import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.chess_fusion_model import ChessFusionAdapter, FusionDecoderLayer, GatedCrossAttention
from training.config import ChessFusionConfig
from training.live_control import TrainingController


class DummyTupleLayer(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return (hidden_states + 0.5,)


class PositionalOnlyXAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_args = None

    def forward(self, *args):
        self.last_args = args
        hidden_states = args[0]
        return hidden_states + 1.0


class DummyDecoderBlock(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return (hidden_states + 0.25,)


class DummyInnerModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderBlock() for _ in range(num_layers)])


class DummyLlamaLikeModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.model = DummyInnerModel(num_layers)


def _make_inputs(
    batch_size: int = 2,
    seq_len: int = 5,
    llm_dim: int = 16,
    perceiver_dim: int = 8,
    context_dim: int = 6,
):
    return {
        "hidden_states": torch.randn(batch_size, seq_len, llm_dim),
        "perceiver_latents": torch.randn(batch_size, 65, perceiver_dim),
        "context": torch.randn(batch_size, 65, context_dim),
        "csmp_square_tokens": torch.randn(batch_size, 64, context_dim),
        "policy_latents": torch.randn(batch_size, 64, perceiver_dim),
        "text_attention_mask": torch.tensor(
            [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]],
            dtype=torch.long,
        ),
    }


@pytest.mark.parametrize("xattn_mode", ["recurrent_query_attn", "structured_square_mixer"])
def test_fusion_decoder_layer_smoke(xattn_mode: str):
    inputs = _make_inputs()
    xattn = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode=xattn_mode,
    )
    wrapper = FusionDecoderLayer(DummyTupleLayer(), xattn)
    wrapper.set_chess_context(
        inputs["perceiver_latents"],
        context=inputs["context"],
        csmp_square_tokens=inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"],
        policy_latents=inputs["policy_latents"],
    )

    outputs = wrapper(inputs["hidden_states"])

    assert isinstance(outputs, tuple)
    assert outputs[0].shape == inputs["hidden_states"].shape
    if xattn_mode == "structured_square_mixer":
        assert xattn._last_structured_metrics is not None
    else:
        assert xattn._last_structured_metrics is None


def test_fusion_decoder_layer_calls_xattn_positionally():
    inputs = _make_inputs()
    xattn = PositionalOnlyXAttn()
    wrapper = FusionDecoderLayer(DummyTupleLayer(), xattn)
    wrapper.set_chess_context(
        inputs["perceiver_latents"],
        context=inputs["context"],
        csmp_square_tokens=inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"],
        policy_latents=inputs["policy_latents"],
    )

    outputs = wrapper(inputs["hidden_states"])

    assert isinstance(outputs, tuple)
    assert outputs[0].shape == inputs["hidden_states"].shape
    assert xattn.last_args is not None
    assert len(xattn.last_args) == 6
    assert xattn.last_args[4] is inputs["text_attention_mask"]
    assert xattn.last_args[5] is inputs["policy_latents"]


def test_pseudotoken_layers_default_to_xattn_layers():
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=True,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=8,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        tap_projection_dim=8,
        perceiver_depth=1,
        perceiver_dim=8,
        perceiver_heads=4,
        num_latents=8,
        xattn_layers=[1, 3],
        xattn_heads=4,
        xattn_recurrent_query_state_dim=8,
        enable_lm_pseudotokens=True,
        num_lm_pseudotokens=2,
        lm_pseudotoken_layers=None,
    )

    adapter = ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)

    assert adapter.pseudotoken_layer_indices == [1, 3]
    assert len(adapter.lm_pseudotoken_layers) == 2


def test_adapter_injects_xattn_and_pseudotokens_on_separate_layer_lists():
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=True,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=8,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        tap_projection_dim=8,
        perceiver_depth=1,
        perceiver_dim=8,
        perceiver_heads=4,
        num_latents=8,
        xattn_layers=[1, 2],
        xattn_heads=4,
        xattn_recurrent_query_state_dim=8,
        enable_lm_pseudotokens=True,
        num_lm_pseudotokens=3,
        lm_pseudotoken_layers=[0, 2],
    )

    adapter = ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)
    llm = DummyLlamaLikeModel(num_layers=4)

    adapter.inject_into_llm(llm)

    layer0 = llm.model.layers[0]
    layer1 = llm.model.layers[1]
    layer2 = llm.model.layers[2]
    layer3 = llm.model.layers[3]

    assert adapter.pseudotoken_layer_indices == [0, 2]
    assert isinstance(layer0, FusionDecoderLayer)
    assert isinstance(layer1, FusionDecoderLayer)
    assert isinstance(layer2, FusionDecoderLayer)
    assert isinstance(layer3, DummyDecoderBlock)

    assert layer0.gated_xattn is None
    assert layer0.pseudotoken_attn is adapter.lm_pseudotoken_layers[0]

    assert layer1.gated_xattn is adapter.gated_xattns[0]
    assert layer1.pseudotoken_attn is None

    assert layer2.gated_xattn is adapter.gated_xattns[1]
    assert layer2.pseudotoken_attn is adapter.lm_pseudotoken_layers[1]

    inputs = _make_inputs(seq_len=4, llm_dim=16, perceiver_dim=8, context_dim=8)
    adapter.set_chess_context(
        inputs["perceiver_latents"],
        context=inputs["context"],
        csmp_square_tokens=inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"],
        policy_latents=inputs["policy_latents"],
    )

    for layer in (layer0, layer1, layer2):
        outputs = layer(inputs["hidden_states"])
        assert isinstance(outputs, tuple)
        assert outputs[0].shape == inputs["hidden_states"].shape


@pytest.mark.parametrize(
    ("missing_field", "expected_message"),
    [
        ("csmp_square_tokens", "requires csmp_square_tokens"),
        ("policy_latents", "requires policy_latents"),
    ],
)
def test_structured_square_mixer_requires_square_aligned_inputs(missing_field: str, expected_message: str):
    inputs = _make_inputs()
    inputs[missing_field] = None

    xattn = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
    )

    with pytest.raises(ValueError, match=expected_message):
        xattn(
            inputs["hidden_states"],
            inputs["perceiver_latents"],
            inputs["context"],
            inputs["csmp_square_tokens"],
            text_attention_mask=inputs["text_attention_mask"],
            policy_latents=inputs["policy_latents"],
        )


def test_structured_square_mixer_metrics_have_expected_shapes_and_normalization():
    inputs = _make_inputs()
    xattn = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
    )

    outputs = xattn(
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"],
        policy_latents=inputs["policy_latents"],
    )

    assert outputs.shape == inputs["hidden_states"].shape
    metrics = xattn._last_structured_metrics
    assert metrics is not None
    assert metrics["router_mode"] == "shared"
    assert metrics["slot_mean"].shape == (192,)
    assert metrics["slot_mean_per_head"].shape == (1, 192)
    assert metrics["square_mean"].shape == (64,)
    assert metrics["square_mean_per_head"].shape == (1, 64)
    assert metrics["source_mass"].shape == (3,)
    assert metrics["source_mass_per_head"].shape == (1, 3)
    assert metrics["global_mean"].shape == (2,)
    assert metrics["global_mean_per_head"].shape == (1, 2)
    assert metrics["effective_gate_abs_mean_per_head"].shape == (4,)
    assert metrics["token_gate_logit_mean_per_head"].shape == (4,)
    torch.testing.assert_close(metrics["slot_mean"].sum(), torch.tensor(1.0), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(metrics["square_mean"].sum(), torch.tensor(1.0), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(metrics["source_mass"].sum(), torch.tensor(1.0), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(metrics["global_mean"].sum(), torch.tensor(1.0), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(metrics["effective_gate_abs_mean"], torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(metrics["token_gate_logit_mean"], torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_structured_square_mixer_last_token_trace_capture_shapes_and_normalization():
    inputs = _make_inputs(batch_size=2, seq_len=4)
    xattn = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
    )
    xattn.eval()
    xattn.set_last_token_trace_capture(True)

    xattn(
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"][:, :4],
        policy_latents=inputs["policy_latents"],
    )

    trace = xattn._last_token_trace
    assert trace is not None
    assert trace["router_mode"] == "shared"
    assert trace["raw_slot_weights"].shape == (2, 192)
    assert trace["source_square_weights"].shape == (2, 3, 64)
    assert trace["aggregate_square_weights"].shape == (2, 64)
    assert trace["global_weights"].shape == (2, 2)
    assert trace["effective_head_gates"].shape == (2, 4)
    assert trace["token_gate_logits"].shape == (2, 4)
    assert trace["source_square_contribution_norms"].shape == (2, 3, 64)
    assert trace["aggregate_square_contribution_norms"].shape == (2, 64)
    assert trace["global_contribution_norms"].shape == (2, 2)
    torch.testing.assert_close(
        trace["raw_slot_weights"].sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["aggregate_square_weights"].sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["global_weights"].sum(dim=-1),
        torch.ones(2),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["source_square_weights"],
        trace["raw_slot_weights"].view(2, 3, 64),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["effective_head_gates"],
        torch.zeros(2, 4),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["aggregate_square_contribution_norms"],
        torch.zeros_like(trace["aggregate_square_contribution_norms"]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["global_contribution_norms"],
        torch.zeros_like(trace["global_contribution_norms"]),
        atol=1e-6,
        rtol=0.0,
    )


def test_structured_square_mixer_last_token_trace_uses_last_valid_token():
    inputs = _make_inputs(batch_size=1, seq_len=4)
    xattn = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
    )
    xattn.eval()
    xattn.set_last_token_trace_capture(True)
    text_attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

    xattn(
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=text_attention_mask,
        policy_latents=inputs["policy_latents"],
    )

    trace = xattn._last_token_trace
    assert trace is not None
    assert int(trace["last_token_indices"][0].item()) == 1

    rq_state = xattn._last_recurrent_query_state
    assert rq_state is not None
    global_perceiver_cond = inputs["perceiver_latents"][:, 64, :].unsqueeze(1).expand(-1, 4, -1)
    square_weight_inputs = xattn.structured_router_stem(torch.cat([rq_state, global_perceiver_cond], dim=-1))
    all_square_weights = torch.softmax(
        xattn.structured_square_weight_proj(square_weight_inputs).float(),
        dim=-1,
    )
    torch.testing.assert_close(
        trace["raw_slot_weights"][0],
        all_square_weights[0, 1],
        atol=1e-5,
        rtol=0.0,
    )


def test_structured_square_mixer_square_weights_condition_on_global_perceiver_latent():
    inputs = _make_inputs(batch_size=1, seq_len=3)
    xattn = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
    )

    square_weight_proj = xattn.structured_square_weight_proj
    assert isinstance(square_weight_proj, nn.Linear)
    assert square_weight_proj.in_features == 16
    xattn.structured_router_stem = nn.Identity()

    with torch.no_grad():
        square_weight_proj.weight.zero_()
        square_weight_proj.bias.zero_()
        square_weight_proj.weight[0, xattn.recurrent_query_state_dim] = 1.0

    baseline_inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    shifted_inputs = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    baseline_inputs["perceiver_latents"][:, 64, :] = 0.0
    shifted_inputs["perceiver_latents"][:, 64, :] = 0.0
    shifted_inputs["perceiver_latents"][:, 64, 0] = 3.0

    xattn(
        baseline_inputs["hidden_states"],
        baseline_inputs["perceiver_latents"],
        baseline_inputs["context"],
        baseline_inputs["csmp_square_tokens"],
        text_attention_mask=baseline_inputs["text_attention_mask"],
        policy_latents=baseline_inputs["policy_latents"],
    )
    baseline_slot_mean = xattn._last_structured_metrics["slot_mean"].clone()

    xattn(
        shifted_inputs["hidden_states"],
        shifted_inputs["perceiver_latents"],
        shifted_inputs["context"],
        shifted_inputs["csmp_square_tokens"],
        text_attention_mask=shifted_inputs["text_attention_mask"],
        policy_latents=shifted_inputs["policy_latents"],
    )
    shifted_slot_mean = xattn._last_structured_metrics["slot_mean"].clone()

    assert not torch.allclose(baseline_slot_mean, shifted_slot_mean)


def test_structured_square_mixer_per_head_trace_capture_shapes_and_normalization():
    inputs = _make_inputs(batch_size=2, seq_len=4)
    xattn = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
        structured_router_mode="per_head",
    )
    xattn.eval()
    xattn.set_last_token_trace_capture(True)

    xattn(
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"][:, :4],
        policy_latents=inputs["policy_latents"],
    )

    trace = xattn._last_token_trace
    assert trace is not None
    assert trace["router_mode"] == "per_head"
    assert trace["raw_slot_weights"].shape == (2, 192)
    assert trace["aggregate_square_weights"].shape == (2, 64)
    assert trace["raw_slot_weights_per_head"].shape == (2, 4, 192)
    assert trace["source_square_weights_per_head"].shape == (2, 4, 3, 64)
    assert trace["aggregate_square_weights_per_head"].shape == (2, 4, 64)
    assert trace["global_weights_per_head"].shape == (2, 4, 2)
    assert trace["effective_head_gates"].shape == (2, 4)
    assert trace["token_gate_logits"].shape == (2, 4)
    assert trace["source_square_contribution_norms"].shape == (2, 3, 64)
    assert trace["aggregate_square_contribution_norms"].shape == (2, 64)
    assert trace["global_contribution_norms"].shape == (2, 2)
    assert trace["source_square_contribution_norms_per_head"].shape == (2, 4, 3, 64)
    assert trace["aggregate_square_contribution_norms_per_head"].shape == (2, 4, 64)
    assert trace["global_contribution_norms_per_head"].shape == (2, 4, 2)
    torch.testing.assert_close(
        trace["raw_slot_weights_per_head"].sum(dim=-1),
        torch.ones(2, 4),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["aggregate_square_weights_per_head"].sum(dim=-1),
        torch.ones(2, 4),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["global_weights_per_head"].sum(dim=-1),
        torch.ones(2, 4),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["raw_slot_weights"],
        trace["raw_slot_weights_per_head"].mean(dim=1),
        atol=1e-5,
        rtol=0.0,
    )


def test_structured_square_mixer_contribution_norm_trace_is_normalized():
    inputs = _make_inputs(batch_size=1, seq_len=4)
    xattn = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
        structured_router_mode="per_head",
        text_gate_mode="none",
    )
    xattn.eval()
    xattn.set_last_token_trace_capture(True)

    with torch.no_grad():
        xattn.gate.fill_(math.atanh(0.5))

    xattn(
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"][:, :4],
        policy_latents=inputs["policy_latents"],
    )

    trace = xattn._last_token_trace
    assert trace is not None
    torch.testing.assert_close(
        trace["aggregate_square_contribution_norms"],
        trace["source_square_contribution_norms"].sum(dim=1),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trace["aggregate_square_contribution_norms_per_head"],
        trace["source_square_contribution_norms_per_head"].sum(dim=2),
        atol=1e-5,
        rtol=0.0,
    )
    total_mass = trace["aggregate_square_contribution_norms"].sum(dim=-1) + trace["global_contribution_norms"].sum(dim=-1)
    per_head_total_mass = (
        trace["aggregate_square_contribution_norms_per_head"].sum(dim=-1)
        + trace["global_contribution_norms_per_head"].sum(dim=-1)
    )
    torch.testing.assert_close(total_mass, torch.ones_like(total_mass), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(per_head_total_mass, torch.ones_like(per_head_total_mass), atol=1e-5, rtol=0.0)


def test_structured_square_mixer_zero_init_tanh_head_matches_none_mode():
    inputs = _make_inputs(batch_size=1, seq_len=4)
    base = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
        text_gate_mode="none",
    )
    gated = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
        text_gate_mode="tanh_head",
    )
    missing, unexpected = gated.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []
    assert sorted(missing) == ["text_gate_mlp.bias", "text_gate_mlp.weight"]
    base.eval()
    gated.eval()
    base.set_last_token_trace_capture(True)
    gated.set_last_token_trace_capture(True)

    base_out = base(
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"][:, :4],
        policy_latents=inputs["policy_latents"],
    )
    gated_out = gated(
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"][:, :4],
        policy_latents=inputs["policy_latents"],
    )

    torch.testing.assert_close(base_out, gated_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        base._last_token_trace["aggregate_square_weights"],
        gated._last_token_trace["aggregate_square_weights"],
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        gated._last_token_trace["token_gate_logits"],
        torch.zeros_like(gated._last_token_trace["token_gate_logits"]),
        atol=1e-6,
        rtol=0.0,
    )


def test_adapter_compute_auxiliary_losses_includes_gate_usage_loss():
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=True,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=8,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        tap_projection_dim=8,
        structured_latents=True,
        num_latents=65,
        perceiver_depth=1,
        perceiver_dim=8,
        perceiver_heads=4,
        use_structured_policy_head=True,
        structured_policy_query_layers=1,
        structured_policy_query_heads=4,
        structured_policy_ffn_mult=2,
        xattn_layers=[0],
        xattn_mode="structured_square_mixer",
        xattn_heads=4,
        xattn_recurrent_query_state_dim=8,
        xattn_text_gate_mode="none",
        aux_policy_weight=0.0,
        structured_xattn_sparse_weight=0.0,
        structured_xattn_square_diversity_weight=0.0,
        structured_xattn_gate_usage_weight=0.6,
        structured_xattn_gate_usage_target=0.4,
        enable_lm_prepend_latents=False,
        enable_lm_pseudotokens=False,
    )
    adapter = ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)
    inputs = _make_inputs(seq_len=5, llm_dim=16, perceiver_dim=8, context_dim=8)

    with torch.no_grad():
        adapter.gated_xattns[0].gate.fill_(math.atanh(0.2))

    adapter.gated_xattns[0](
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"],
        policy_latents=inputs["policy_latents"],
    )

    losses = adapter.compute_auxiliary_losses(
        policy_logits=torch.zeros(inputs["hidden_states"].size(0), 8),
        eval_logits=None,
        policy_targets=None,
    )

    expected_usage = torch.tensor(0.2)
    expected_loss = torch.tensor(0.2)
    torch.testing.assert_close(
        losses["structured_xattn_gate_usage_mean_abs"],
        expected_usage,
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        losses["structured_xattn_gate_usage_loss"],
        expected_loss,
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        losses["total_aux_loss"],
        cfg.structured_xattn_gate_usage_weight * expected_loss,
        atol=1e-5,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("use_csmp", "use_policy_head", "expected_message"),
    [
        (False, True, "requires use_chess_structure_mp=True"),
        (True, False, "requires use_structured_policy_head=True"),
    ],
)
def test_structured_square_mixer_adapter_validation(use_csmp: bool, use_policy_head: bool, expected_message: str):
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=use_csmp,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=8,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        tap_projection_dim=8,
        structured_latents=True,
        num_latents=65,
        perceiver_dim=8,
        perceiver_heads=4,
        use_structured_policy_head=use_policy_head,
        structured_policy_query_layers=1,
        structured_policy_query_heads=4,
        xattn_layers=[0],
        xattn_mode="structured_square_mixer",
        xattn_heads=4,
        xattn_recurrent_query_state_dim=8,
    )

    with pytest.raises(ValueError, match=expected_message):
        ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)


def test_structured_square_mixer_skips_legacy_recurrent_query_projection_params():
    xattn = GatedCrossAttention(
        llm_dim=16,
        perceiver_dim=8,
        context_dim=6,
        n_heads=4,
        recurrent_query_state_dim=8,
        xattn_mode="structured_square_mixer",
    )

    assert xattn.q_norm is None
    assert xattn.q_proj is None
    assert xattn.k_proj is None
    assert xattn.v_proj is None
    assert xattn.recurrent_query_proj is None

    param_names = set(dict(xattn.named_parameters()).keys())
    assert "recurrent_query_proj.weight" not in param_names
    assert "recurrent_query_proj.bias" not in param_names
    assert "q_proj.weight" not in param_names
    assert "k_proj.weight" not in param_names
    assert "v_proj.weight" not in param_names


def test_structured_square_mixer_warns_when_shared_readout_settings_are_ignored(capsys: pytest.CaptureFixture[str]):
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=True,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=8,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        tap_projection_dim=8,
        structured_latents=True,
        num_latents=65,
        perceiver_depth=1,
        perceiver_dim=8,
        perceiver_heads=4,
        use_structured_policy_head=True,
        structured_policy_query_layers=1,
        structured_policy_query_heads=4,
        structured_policy_ffn_mult=2,
        xattn_layers=[0],
        xattn_mode="structured_square_mixer",
        xattn_heads=4,
        xattn_recurrent_query_state_dim=8,
        num_fusion_tokens=24,
        readout_depth=2,
        readout_use_policy_latent_cross_attention=True,
        enable_lm_prepend_latents=False,
        enable_lm_pseudotokens=False,
    )

    ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)
    out = capsys.readouterr().out

    assert "structured_square_mixer' ignores shared readout settings" in out
    assert "num_fusion_tokens=24" in out
    assert "readout_depth=2" in out
    assert "readout_use_policy_latent_cross_attention=True" in out


def test_adapter_exposes_named_structured_xattn_metrics():
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=True,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=8,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        tap_projection_dim=8,
        structured_latents=True,
        num_latents=65,
        perceiver_depth=1,
        perceiver_dim=8,
        perceiver_heads=4,
        use_structured_policy_head=True,
        structured_policy_query_layers=1,
        structured_policy_query_heads=4,
        structured_policy_ffn_mult=2,
        xattn_layers=[0],
        xattn_mode="structured_square_mixer",
        xattn_heads=4,
        xattn_recurrent_query_state_dim=8,
        enable_lm_prepend_latents=False,
        enable_lm_pseudotokens=False,
    )
    adapter = ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)
    inputs = _make_inputs(seq_len=5, llm_dim=16, perceiver_dim=8, context_dim=8)

    adapter.gated_xattns[0](
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"],
        policy_latents=inputs["policy_latents"],
    )

    metrics = adapter.get_structured_xattn_metrics()
    assert "structured_xattn/layer_0/slot_mean_std" in metrics
    assert "structured_xattn/layer_0/square_entropy_norm" in metrics
    assert "structured_xattn/layer_0/square_usage_entropy_norm" in metrics
    assert "structured_xattn/layer_0/max_square_index" in metrics
    assert "structured_xattn/layer_0/slot_mean_min" in metrics
    assert "structured_xattn/layer_0/slot_mean_max" in metrics
    assert "structured_xattn/layer_0/square_mean_std" in metrics
    assert "structured_xattn/layer_0/csmp/slot_mean_std" in metrics
    assert "structured_xattn/layer_0/perceiver/slot_max_square_index" in metrics
    assert "structured_xattn/layer_0/source_mass/policy" in metrics
    assert "structured_xattn/layer_0/global_mass/side_token" in metrics
    assert "structured_xattn/layer_0/effective_gate_abs_mean" in metrics
    assert "structured_xattn/layer_0/token_gate_logit_mean" in metrics


def test_adapter_compute_auxiliary_losses_includes_structured_xattn_sparse_loss():
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=True,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=8,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        tap_projection_dim=8,
        structured_latents=True,
        num_latents=65,
        perceiver_depth=1,
        perceiver_dim=8,
        perceiver_heads=4,
        use_structured_policy_head=True,
        structured_policy_query_layers=1,
        structured_policy_query_heads=4,
        structured_policy_ffn_mult=2,
        xattn_layers=[0],
        xattn_mode="structured_square_mixer",
        xattn_heads=4,
        xattn_recurrent_query_state_dim=8,
        aux_policy_weight=0.0,
        structured_xattn_sparse_weight=0.25,
        enable_lm_prepend_latents=False,
        enable_lm_pseudotokens=False,
    )
    adapter = ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)
    inputs = _make_inputs(seq_len=5, llm_dim=16, perceiver_dim=8, context_dim=8)

    adapter.gated_xattns[0](
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"],
        policy_latents=inputs["policy_latents"],
    )

    expected_sparse_loss = adapter.gated_xattns[0]._last_structured_square_sparse_loss.detach().clone()
    losses = adapter.compute_auxiliary_losses(
        policy_logits=torch.zeros(inputs["hidden_states"].size(0), 8),
        eval_logits=None,
        policy_targets=None,
    )

    torch.testing.assert_close(losses["structured_xattn_sparse_loss"], expected_sparse_loss, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        losses["total_aux_loss"],
        cfg.structured_xattn_sparse_weight * expected_sparse_loss,
        atol=1e-6,
        rtol=0.0,
    )


def test_adapter_compute_auxiliary_losses_includes_square_diversity_loss():
    cfg = ChessFusionConfig(
        use_cnn=False,
        use_transformer_taps=False,
        use_chess_structure_mp=True,
        csmp_layers=1,
        csmp_heads=8,
        csmp_dim=8,
        csmp_pos_dim=4,
        csmp_piece_dim=4,
        tap_projection_dim=8,
        structured_latents=True,
        num_latents=65,
        perceiver_depth=1,
        perceiver_dim=8,
        perceiver_heads=4,
        use_structured_policy_head=True,
        structured_policy_query_layers=1,
        structured_policy_query_heads=4,
        structured_policy_ffn_mult=2,
        xattn_layers=[0],
        xattn_mode="structured_square_mixer",
        xattn_heads=4,
        xattn_recurrent_query_state_dim=8,
        aux_policy_weight=0.0,
        structured_xattn_sparse_weight=0.0,
        structured_xattn_square_diversity_weight=0.4,
        structured_xattn_square_diversity_target_entropy=0.7,
        enable_lm_prepend_latents=False,
        enable_lm_pseudotokens=False,
    )
    adapter = ChessFusionAdapter(SimpleNamespace(chess_fusion=cfg), llm_dim=16, llm_num_heads=4)
    inputs = _make_inputs(seq_len=5, llm_dim=16, perceiver_dim=8, context_dim=8)

    adapter.gated_xattns[0](
        inputs["hidden_states"],
        inputs["perceiver_latents"],
        inputs["context"],
        inputs["csmp_square_tokens"],
        text_attention_mask=inputs["text_attention_mask"],
        policy_latents=inputs["policy_latents"],
    )

    expected_usage_entropy = adapter.gated_xattns[0]._last_structured_square_usage_entropy_norm.detach().clone()
    expected_diversity_loss = torch.clamp_min(
        torch.tensor(cfg.structured_xattn_square_diversity_target_entropy) - expected_usage_entropy,
        0.0,
    )
    losses = adapter.compute_auxiliary_losses(
        policy_logits=torch.zeros(inputs["hidden_states"].size(0), 8),
        eval_logits=None,
        policy_targets=None,
    )

    torch.testing.assert_close(
        losses["structured_xattn_square_usage_entropy"],
        expected_usage_entropy,
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        losses["structured_xattn_square_diversity_loss"],
        expected_diversity_loss,
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        losses["total_aux_loss"],
        cfg.structured_xattn_square_diversity_weight * expected_diversity_loss,
        atol=1e-6,
        rtol=0.0,
    )


def test_training_controller_round_trips_structured_xattn_sparse_weight(monkeypatch: pytest.MonkeyPatch):
    cfg = ChessFusionConfig(
        structured_xattn_sparse_weight=0.15,
        structured_xattn_square_diversity_weight=0.2,
        structured_xattn_square_diversity_target_entropy=0.65,
        structured_xattn_gate_usage_weight=0.12,
        structured_xattn_gate_usage_target=0.3,
    )
    controller = TrainingController(output_dir="unused-test-dir", poll_steps=1)
    config = SimpleNamespace(
        learning_rate=1e-4,
        model=SimpleNamespace(
            mode="chess_fusion",
            enable_lm=True,
            chess_fusion=cfg,
        ),
    )
    fake_state = {}
    fake_mtime = {"value": 0.0}

    def fake_read_state():
        return copy.deepcopy(fake_state)

    def fake_write_state(state):
        fake_state.clear()
        fake_state.update(copy.deepcopy(state))
        fake_mtime["value"] += 1.0

    monkeypatch.setattr(Path, "mkdir", lambda self, parents=False, exist_ok=False: None)
    monkeypatch.setattr(controller, "_read_state", fake_read_state)
    monkeypatch.setattr(controller, "_write_state", fake_write_state)
    monkeypatch.setattr("training.live_control.os.path.getmtime", lambda _: fake_mtime["value"])

    state = controller.init_control_file(config)
    assert state["status"]["active_structured_xattn_sparse_weight"] == pytest.approx(0.15)
    assert state["status"]["active_structured_xattn_square_diversity_weight"] == pytest.approx(0.2)
    assert state["status"]["active_structured_xattn_square_diversity_target_entropy"] == pytest.approx(0.65)
    assert state["status"]["active_structured_xattn_gate_usage_weight"] == pytest.approx(0.12)
    assert state["status"]["active_structured_xattn_gate_usage_target"] == pytest.approx(0.3)

    updated_state = controller._read_state()
    updated_state["structured_xattn_sparse_weight"] = 0.35
    updated_state["structured_xattn_square_diversity_weight"] = 0.45
    updated_state["structured_xattn_square_diversity_target_entropy"] = 0.8
    updated_state["structured_xattn_gate_usage_weight"] = 0.25
    updated_state["structured_xattn_gate_usage_target"] = 0.55
    controller._write_state(updated_state)

    changes = controller.poll()

    assert changes is not None
    assert changes["structured_xattn_sparse_weight"] == pytest.approx(0.35)
    assert changes["structured_xattn_square_diversity_weight"] == pytest.approx(0.45)
    assert changes["structured_xattn_square_diversity_target_entropy"] == pytest.approx(0.8)
    assert changes["structured_xattn_gate_usage_weight"] == pytest.approx(0.25)
    assert changes["structured_xattn_gate_usage_target"] == pytest.approx(0.55)
    assert controller._read_state()["structured_xattn_sparse_weight"] is None
    assert controller._read_state()["structured_xattn_square_diversity_weight"] is None
    assert controller._read_state()["structured_xattn_square_diversity_target_entropy"] is None
    assert controller._read_state()["structured_xattn_gate_usage_weight"] is None
    assert controller._read_state()["structured_xattn_gate_usage_target"] is None
