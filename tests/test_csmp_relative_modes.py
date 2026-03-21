from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.chess_fusion_model import ChessFusionAdapter
from training.chess_structure_mp import ChessStructureAttention
from training.config import ChessFusionConfig


def _copy_attention_core(src: ChessStructureAttention, dst: ChessStructureAttention) -> None:
    dst.q_proj.load_state_dict(src.q_proj.state_dict())
    dst.k_proj.load_state_dict(src.k_proj.state_dict())
    dst.v_proj.load_state_dict(src.v_proj.state_dict())
    dst.o_proj.load_state_dict(src.o_proj.state_dict())


def _set_zero_projection(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()


def _set_identity_projection(linear: nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight.copy_(torch.eye(linear.out_features, linear.in_features))
        if linear.bias is not None:
            linear.bias.zero_()


def _build_sparse_head_masks(batch_size: int = 1, num_heads: int = 8) -> torch.Tensor:
    mask = torch.zeros(batch_size, num_heads, 64, 64, dtype=torch.bool)
    sq_idx = torch.arange(64)
    mask[:, :, sq_idx, sq_idx] = True
    mask[:, :, 0, 0] = False
    mask[:, :, 0, 1] = True
    mask[:, :, 0, 8] = True
    return mask


def _build_minimal_boards(batch_size: int = 2) -> torch.Tensor:
    boards = torch.zeros(batch_size, 18, 8, 8)
    boards[:, 5, 0, 4] = 1.0   # White king on e1
    boards[:, 11, 7, 4] = 1.0  # Black king on e8
    return boards


def test_chess_fusion_config_relative_mode_defaults_and_validation():
    cfg = ChessFusionConfig()
    assert cfg.csmp_relative_mode == "none"
    assert cfg.csmp_relative_edge_dim == 16

    with pytest.raises(ValueError, match="csmp_relative_mode"):
        ChessFusionConfig(csmp_relative_mode="bad_mode")

    with pytest.raises(ValueError, match="csmp_relative_edge_dim"):
        ChessFusionConfig(csmp_relative_edge_dim=0)


def test_adapter_threads_csmp_relative_config_into_chess_mp():
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=True,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=16,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        csmp_relative_mode="score_bias",
        csmp_relative_edge_dim=24,
        tap_projection_dim=16,
        perceiver_depth=1,
        perceiver_dim=16,
        perceiver_heads=4,
    )
    adapter = ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)

    assert adapter.multi_scale.chess_mp.relative_mode == "score_bias"
    assert adapter.multi_scale.chess_mp.relative_edge_dim == 24
    assert adapter.multi_scale.chess_mp.layers[0].attn.relative_mode == "score_bias"
    assert adapter.multi_scale.chess_mp.layers[0].attn.relative_edge_dim == 24


def test_csmp_relative_modes_match_baseline_when_new_params_are_zero():
    torch.manual_seed(0)
    base = ChessStructureAttention(dim=16, n_heads=8, dropout=0.0, relative_mode="none")
    score = ChessStructureAttention(dim=16, n_heads=8, dropout=0.0, relative_mode="score_bias")
    edge = ChessStructureAttention(
        dim=16,
        n_heads=8,
        dropout=0.0,
        relative_mode="edge_modulation",
        relative_edge_dim=8,
    )
    _copy_attention_core(base, score)
    _copy_attention_core(base, edge)

    x = torch.randn(2, 64, 16)
    head_masks = torch.ones(2, 8, 64, 64, dtype=torch.bool)

    base_out = base(x, head_masks=head_masks)
    score_out = score(x, head_masks=head_masks)
    edge_out = edge(x, head_masks=head_masks)

    torch.testing.assert_close(base_out, score_out, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(base_out, edge_out, atol=1e-6, rtol=1e-5)


def test_score_bias_prefers_allowed_relative_offsets_without_unmasking_blocked_edges():
    attn = ChessStructureAttention(dim=8, n_heads=8, dropout=0.0, relative_mode="score_bias")
    _set_zero_projection(attn.q_proj)
    _set_zero_projection(attn.k_proj)
    _set_identity_projection(attn.v_proj)
    _set_identity_projection(attn.o_proj)

    x = torch.zeros(1, 64, 8)
    x[0, 1, 0] = 1.0
    x[0, 8, 0] = 3.0
    x[0, 2, 0] = 9.0  # masked for query square 0; should stay ignored
    head_masks = _build_sparse_head_masks()

    base_out = attn(x, head_masks=head_masks)
    torch.testing.assert_close(base_out[0, 0, 0], torch.tensor(2.0), atol=1e-6, rtol=0.0)

    with torch.no_grad():
        attn.relative_score_bias.zero_()
        attn.relative_score_bias[0, 8, 7] = 4.0   # query a1 -> key a2 : delta (+1 rank, 0 file)
        attn.relative_score_bias[0, 7, 9] = 20.0  # query a1 -> key c1 : masked, must remain ignored

    biased_out = attn(x, head_masks=head_masks)
    expected_weights = torch.softmax(torch.tensor([0.0, 4.0]), dim=0)
    expected_value = expected_weights[0] * 1.0 + expected_weights[1] * 3.0
    torch.testing.assert_close(biased_out[0, 0, 0], expected_value, atol=1e-5, rtol=0.0)
    assert biased_out[0, 0, 0].item() < 4.0


def test_edge_modulation_changes_pairwise_messages_while_hard_masks_remain_active():
    attn = ChessStructureAttention(
        dim=8,
        n_heads=8,
        dropout=0.0,
        relative_mode="edge_modulation",
        relative_edge_dim=4,
    )
    _set_identity_projection(attn.q_proj)
    _set_zero_projection(attn.k_proj)
    _set_identity_projection(attn.v_proj)
    _set_identity_projection(attn.o_proj)

    x = torch.zeros(1, 64, 8)
    x[0, 0, 0] = 1.0
    x[0, 1, 0] = 1.0
    x[0, 8, 0] = 3.0
    x[0, 2, 0] = 9.0  # masked for query square 0; should stay ignored
    head_masks = _build_sparse_head_masks()

    with torch.no_grad():
        attn.edge_rank_embedding.weight.zero_()
        attn.edge_file_embedding.weight.zero_()
        attn.edge_head_embedding.weight.zero_()
        attn.edge_proj.weight.zero_()
        attn.edge_proj.bias.zero_()
        attn.edge_to_k.weight.zero_()
        attn.edge_to_k.bias.zero_()
        attn.edge_to_v.weight.zero_()
        attn.edge_to_v.bias.zero_()

        # Only delta_rank=+1 edges emit a non-zero edge embedding. For query a1, this
        # targets a2 (square 8), while the masked c1 edge remains irrelevant.
        attn.edge_rank_embedding.weight[8, 0] = 1.0
        attn.edge_proj.weight[0, 0] = 1.0
        attn.edge_to_k.weight[0, 0] = 2.0
        attn.edge_to_v.weight[1, 0] = 1.0  # beta channel only

    modulated_out = attn(x, head_masks=head_masks)
    expected_weights = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)
    expected_value = expected_weights[0] * 1.0 + expected_weights[1] * 4.0
    torch.testing.assert_close(modulated_out[0, 0, 0], expected_value, atol=1e-5, rtol=0.0)
    assert modulated_out[0, 0, 0].item() < 5.0


@pytest.mark.parametrize("relative_mode", ["none", "score_bias", "edge_modulation"])
def test_chess_fusion_adapter_board_only_smoke_for_relative_modes(relative_mode: str):
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=True,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=16,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        csmp_relative_mode=relative_mode,
        csmp_relative_edge_dim=12,
        tap_projection_dim=16,
        perceiver_depth=1,
        perceiver_dim=16,
        perceiver_heads=4,
        num_latents=8,
        enable_lm_prepend_latents=False,
        enable_lm_pseudotokens=False,
    )
    adapter = ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)
    boards = _build_minimal_boards(batch_size=2)
    side_to_move = torch.tensor([1, 0], dtype=torch.long)

    out = adapter(boards, side_to_move=side_to_move)

    assert out["csmp_square_tokens"].shape == (2, 64, 16)
    assert out["perceiver_latents"].shape == (2, 8, 16)
    assert out["csmp_square_tokens"].dtype == boards.dtype
    assert out["perceiver_latents"].dtype == boards.dtype
