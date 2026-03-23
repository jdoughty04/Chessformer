"""
Chess-Fusion: Deep Cross-Modal Chess-LLM Integration.

This module provides:
1. MultiScaleFeatureExtractor - Taps CNN, mid-transformer, and final transformer features
2. SquareLatentEncoder - Perceiver with dual attention-pooled branches (aux + fusion)
3. SharedLayerReadout - Single readout conditioned on layer fraction via AdaLN
4. GatedCrossAttention - tanh-gated cross-attention for injection into LLM layers
5. FusionDecoderLayer - Wrapper that adds gated cross-attention after an LLM layer
6. ChessFusionAdapter - Top-level adapter combining all components
"""

import sys
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def _compiler_is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return False
    return bool(compiler.is_compiling())


def _compiler_disable(fn):
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        return fn
    return compiler.disable(fn)

from training.maia_model import MaiaPolicyModel, elo_to_category
from training.chess_structure_mp import ChessStructureMP, DynamicMaskBuilder, extract_piece_types

try:
    from training.maia_model import get_maia_mapping
except Exception:
    get_maia_mapping = None

_ABS_SQUARE_NAMES: Tuple[str, ...] = tuple(
    f"{chr(ord('a') + file_idx)}{rank_idx + 1}"
    for rank_idx in range(8)
    for file_idx in range(8)
)


# =============================================================================
# ManualMultiHeadAttention â€” drop-in SDPA replacement for nn.MultiheadAttention
# =============================================================================

class ManualMultiHeadAttention(nn.Module):
    """
    Drop-in replacement for ``nn.MultiheadAttention`` that uses
    ``F.scaled_dot_product_attention`` internally.

    Key differences from ``nn.MultiheadAttention``:
    - Stores Q/K/V/O as separate ``nn.Linear`` layers (no packed in_proj_weight).
    - Uses SDPA, enabling Flash Attention / memory-efficient attention dispatch.
    - Masks follow **nn.MHA convention** at the call site (True = masked-out)
      and are inverted internally before passing to SDPA (True = attend).

    The ``need_weights=True`` path falls back to a manual matmul+softmax
    implementation so that attention weights can be returned for entropy
    logging.  This path is guarded by ``torch.compiler.is_compiling()``
    inside callers so torch.compile never traces it.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        kdim = kdim if kdim is not None else embed_dim
        vdim = vdim if vdim is not None else embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (B, Q, embed_dim)
            key:   (B, K, kdim)
            value: (B, K, vdim)
            attn_mask: bool tensor in **nn.MHA convention** (True = masked-out).
                       Accepted shapes: (Q, K) or (B*H, Q, K).
            need_weights: if True, return (output, weights) via manual softmax.
            average_attn_weights: if True and need_weights, average over heads.

        Returns:
            output: (B, Q, embed_dim)
            attn_weights: (B, H, Q, K) if need_weights and not average_attn_weights,
                          (B, Q, K) if need_weights and average_attn_weights,
                          None otherwise.
        """
        B, Q_len, _ = query.shape
        K_len = key.size(1)
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, Q_len, H, D).transpose(1, 2)  # (B, H, Q, D)
        k = self.k_proj(key).view(B, K_len, H, D).transpose(1, 2)    # (B, H, K, D)
        v = self.v_proj(value).view(B, K_len, H, D).transpose(1, 2)  # (B, H, K, D)

        # --- Mask handling ---
        # Call-site masks use nn.MHA convention: True = masked-out.
        # SDPA bool masks: True = attend.  Invert here.
        sdpa_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                sdpa_mask = ~attn_mask  # invert
            else:
                # Float mask (e.g. -inf for masked positions) â€” pass through
                sdpa_mask = attn_mask

            # Expand 2D (Q, K) â†’ (1, 1, Q, K) for broadcasting
            if sdpa_mask.dim() == 2:
                sdpa_mask = sdpa_mask.unsqueeze(0).unsqueeze(0)
            elif sdpa_mask.dim() == 3:
                # (B*H, Q, K) â†’ (B, H, Q, K)
                sdpa_mask = sdpa_mask.view(B, H, Q_len, K_len)

        if need_weights:
            # Manual path â€” needed for entropy logging.
            # Not traced by torch.compile (callers guard with is_compiling()).
            scale = 1.0 / math.sqrt(float(D))
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, Q, K)
            if sdpa_mask is not None:
                if sdpa_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~sdpa_mask, float('-inf'))
                else:
                    scores = scores + sdpa_mask
            weights = torch.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0.0:
                weights = F.dropout(weights, p=self.dropout_p)
            out = torch.matmul(weights, v)  # (B, H, Q, D)
            out = out.transpose(1, 2).contiguous().view(B, Q_len, self.embed_dim)
            out = self.out_proj(out)
            if average_attn_weights:
                weights = weights.mean(dim=1)  # (B, Q, K)
            return out, weights
        else:
            if hasattr(F, "scaled_dot_product_attention"):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=sdpa_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                )  # (B, H, Q, D)
            else:
                # Compatibility fallback for older torch versions without SDPA.
                scale = 1.0 / math.sqrt(float(D))
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                if sdpa_mask is not None:
                    if sdpa_mask.dtype == torch.bool:
                        scores = scores.masked_fill(~sdpa_mask, float('-inf'))
                    else:
                        scores = scores + sdpa_mask
                weights = torch.softmax(scores, dim=-1)
                if self.training and self.dropout_p > 0.0:
                    weights = F.dropout(weights, p=self.dropout_p)
                out = torch.matmul(weights, v)
            out = out.transpose(1, 2).contiguous().view(B, Q_len, self.embed_dim)
            out = self.out_proj(out)
            return out, None


# =============================================================================
# Perspective Un-Mirroring Utilities
# =============================================================================

@torch.no_grad()
def unmirror_board_tensor(boards: torch.Tensor, side_to_move: torch.Tensor) -> torch.Tensor:
    """Convert perspective-relative Maia2 board tensors to absolute coordinates.

    Maia2's extract_maia_features() mirrors the board when Black is to move:
    - Spatial vertical flip (ranks reversed)
    - Piece color swap: channels 0-5 become "our pieces", 6-11 "their pieces"
    - Side-to-move channel (12) set to 1 (White)
    - Castling rights colors swapped (13-14 â†” 15-16)

    This reverses that transformation so all positions use a consistent
    absolute coordinate system (a1 = square 0, White pieces in channels 0-5).

    Args:
        boards: (B, 18, 8, 8) perspective-relative board tensors
        side_to_move: (B,) long tensor, 1=White, 0=Black

    Returns:
        (B, 18, 8, 8) board tensors in absolute coordinates
    """
    black_mask = (side_to_move == 0)
    if not black_mask.any():
        return boards

    abs_boards = boards.clone()
    b = abs_boards[black_mask].flip(-2)  # spatial un-flip (reverse vertical mirror)

    # Swap piece channels (0-5 â†” 6-11), fix side-to-move, swap castling rights
    abs_boards[black_mask] = torch.cat([
        b[:, 6:12],     # White pieces â†’ ch 0-5  (were in 6-11 after mirror color swap)
        b[:, 0:6],      # Black pieces â†’ ch 6-11 (were in 0-5 after mirror color swap)
        torch.zeros_like(b[:, 12:13]),  # side-to-move = 0 (Black to move)
        b[:, 15:16],    # White K-side castling â†’ ch 13 (was at ch 15 after mirror)
        b[:, 16:17],    # White Q-side castling â†’ ch 14 (was at ch 16 after mirror)
        b[:, 13:14],    # Black K-side castling â†’ ch 15 (was at ch 13 after mirror)
        b[:, 14:15],    # Black Q-side castling â†’ ch 16 (was at ch 14 after mirror)
        b[:, 17:18],    # en passant (spatial already fixed by flip)
    ], dim=1)

    return abs_boards


def _unmirror_cnn_features(
    cnn_outputs: List[torch.Tensor],
    side_to_move: torch.Tensor,
) -> List[torch.Tensor]:
    """Un-mirror CNN spatial features for Black-to-move samples.

    When Black is to move, the board was mirrored before CNN processing,
    so CNN features are in perspective-relative spatial coordinates.
    This flips the rank dimension back to absolute board coordinates.

    Args:
        cnn_outputs: list of (B, C, 8, 8) tensors from CNN hooks
        side_to_move: (B,) long tensor, 1=White, 0=Black

    Returns:
        list of (B, C, 8, 8) tensors in absolute spatial coordinates
    """
    black_mask = (side_to_move == 0)
    if not black_mask.any():
        return cnn_outputs

    mask_4d = black_mask.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for broadcasting
    return [
        torch.where(mask_4d, cnn_out.flip(-2), cnn_out)
        for cnn_out in cnn_outputs
    ]


# =============================================================================
# Multi-Scale Feature Extractor
# =============================================================================

class MultiScaleFeatureExtractor(nn.Module):
    """
    Hooks into Maia2 backbone to extract features at multiple scales.

    Maia2 CNN architecture (ChessResNet):
      conv1+bn1+relu  â†’ (B, 256, 8, 8)     initial projection
      layers          â†’ Sequential of 5 BasicBlock residual blocks (indices 0-4)
                         each outputs (B, 256, 8, 8)
      conv_last+bn_last â†’ (B, 8, 8, 8)     compress to 8 channels
      view(B, 8, 64)  â†’ to_patch_embedding â†’ (B, 8, 1024)  8 channel-tokens
      transformer     â†’ (B, 8, 1024)       8 channel-tokens

    We extract:
      CNN taps: For each block index in cnn_tap_layers, hook that BasicBlock's
               output â†’ (B, 256, 8, 8) â†’ reshape (B, 64, 256) as spatial tokens
               + shared learned rank/file positional encoding (16 dims)
               â†’ per-tap projection â†’ (B, 64, tap_dim)
               Total CNN tokens: len(cnn_tap_layers) * 64

      L_mid  : transformer elo_layers[0].ff â†’ (B, 8, 1024) â†’ project â†’ (B, 8, tap_dim)
               [optional, controlled by use_transformer_taps]
      L_final: transformer output (after norm) â†’ (B, 8, 1024) â†’ project â†’ (B, 8, tap_dim)
               [optional, controlled by use_transformer_taps]
      side   : learned side-to-move embedding â†’ (B, 1, tap_dim)

    Output: concat â†’ (B, N_cnn*64 + [8+8] + 1, tap_dim)
    """

    def __init__(self, maia_backbone: Optional[MaiaPolicyModel], tap_dim: int = 1024,
                 cnn_tap_layers: List[int] = None,
                 concat_cnn_taps: bool = False,
                 use_transformer_taps: bool = True,
                 use_cnn: bool = True,
                 csmp_config: Optional[Dict] = None):
        super().__init__()
        self.maia = maia_backbone
        self.tap_dim = tap_dim
        self.concat_cnn_taps = concat_cnn_taps
        self.use_cnn = use_cnn
        self.use_transformer_taps = use_transformer_taps and use_cnn  # transformer taps require backbone
        self.use_csmp = csmp_config is not None

        # Hook storage
        self._cnn_outputs: Dict[int, torch.Tensor] = {}
        self._mid_output = None
        self._cached_cnn_final: Optional[torch.Tensor] = None
        self._hooks = []

        if use_cnn:
            # Maia dims
            self.cnn_dim = self.maia.maia.cfg.dim_cnn     # 256
            self.vit_dim = self.maia.maia.cfg.dim_vit      # 1024
            self.num_cnn_blocks = self.maia.maia.cfg.num_blocks_cnn  # 5

            # Validate and store CNN tap layer indices
            if cnn_tap_layers is None:
                cnn_tap_layers = [self.num_cnn_blocks - 1]  # default: final block only
            for idx in cnn_tap_layers:
                if idx < 0 or idx >= self.num_cnn_blocks:
                    raise ValueError(
                        f"cnn_tap_layers index {idx} out of range [0, {self.num_cnn_blocks - 1}]"
                    )
            self.cnn_tap_layers = sorted(cnn_tap_layers)
            self.num_cnn_taps = len(self.cnn_tap_layers)
        else:
            self.cnn_dim = 256  # unused but keep for type consistency
            self.vit_dim = 1024
            self.num_cnn_blocks = 0
            self.cnn_tap_layers = []
            self.num_cnn_taps = 0

        # --- CSMP or legacy CNN projection ---
        if self.use_csmp:
            # Chess Structure Message Passing replaces CNN projection + pos_enc
            cnn_proj = csmp_config.get('csmp_cnn_proj_dim', 0)
            self.chess_mp = ChessStructureMP(
                cnn_dim=self.cnn_dim if use_cnn else 256,
                num_taps=self.num_cnn_taps,  # 0 when CNN disabled â†’ pos+piece only
                output_dim=tap_dim,
                csmp_dim=csmp_config.get('csmp_dim', 1024),
                pos_dim=csmp_config.get('csmp_pos_dim', 32),
                piece_dim=csmp_config.get('csmp_piece_dim', 64),
                cnn_proj_dim=cnn_proj if cnn_proj > 0 else None,
                n_layers=csmp_config.get('csmp_layers', 4),
                n_heads=csmp_config.get('csmp_heads', 8),
                ffn_mult=csmp_config.get('csmp_ffn_mult', 2),
                dropout=csmp_config.get('csmp_dropout', 0.1),
                use_ray_mask=csmp_config.get('csmp_use_ray_mask', True),
                use_attack_mask=csmp_config.get('csmp_use_attack_mask', True),
                use_xy_coords=csmp_config.get('csmp_use_xy_coords', False),
                relative_mode=csmp_config.get('csmp_relative_mode', 'none'),
                relative_edge_dim=csmp_config.get('csmp_relative_edge_dim', 16),
                ablation_no_mask=csmp_config.get('csmp_ablation_no_mask', False),
            )
        elif use_cnn:
            # Legacy path: shared positional encoding + direct projection
            self.pos_enc_dim = 16  # rank(8 embed) + file(8 embed)
            self.rank_embedding = nn.Embedding(8, 8)
            self.file_embedding = nn.Embedding(8, 8)

            # --- CNN projection ---
            if concat_cnn_taps:
                # Concat mode: all taps per square â†’ single projection
                concat_input_dim = self.num_cnn_taps * self.cnn_dim + self.pos_enc_dim
                self.proj_cnn_concat = nn.Sequential(
                    nn.Linear(concat_input_dim, tap_dim),
                    nn.LayerNorm(tap_dim),
                )
            else:
                # Per-tap mode: separate projection per tap layer
                self.proj_cnn = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.cnn_dim + self.pos_enc_dim, tap_dim),
                        nn.LayerNorm(tap_dim),
                    )
                    for _ in self.cnn_tap_layers
                ])
        else:
            raise ValueError("use_cnn=False requires use_chess_structure_mp=True (CSMP handles board-only encoding)")

        # --- Transformer tap projections (optional, requires backbone) ---
        if self.use_transformer_taps:
            self.proj_mid = nn.Sequential(
                nn.Linear(self.vit_dim, tap_dim),
                nn.LayerNorm(tap_dim),
            )
            self.proj_final = nn.Sequential(
                nn.Linear(self.vit_dim, tap_dim),
                nn.LayerNorm(tap_dim),
            )

        # --- Side-to-move context token ---
        self.side_token = nn.Embedding(2, tap_dim)

        # Register hooks (only when CNN is active)
        if use_cnn:
            self._register_hooks()

        # Log context size
        if use_cnn:
            cnn_tokens = 64 if concat_cnn_taps else self.num_cnn_taps * 64
        else:
            cnn_tokens = 64  # CSMP still produces 64 spatial tokens from pos+piece
        n_ctx = cnn_tokens + (16 if self.use_transformer_taps else 0) + 1
        mode_label = "CNN-free (pos+piece only)" if not use_cnn else (
            "concat" if concat_cnn_taps else f"{self.num_cnn_taps}x64"
        )
        print(f"  [MultiScale] mode={mode_label}, "
              f"transformer_taps={self.use_transformer_taps}, "
              f"total context tokens={n_ctx}")

    def _register_hooks(self):
        """Register forward hooks on individual CNN blocks and optionally mid-transformer."""
        cnn_sequential = self.maia.maia.chess_cnn.layers

        # Hook each requested CNN block
        for block_idx in self.cnn_tap_layers:
            block = cnn_sequential[block_idx]

            def make_cnn_hook(idx):
                def hook(module, input, output):
                    self._cnn_outputs[idx] = output
                return hook

            self._hooks.append(
                block.register_forward_hook(make_cnn_hook(block_idx))
            )
            print(f"  [MultiScale] Hooked CNN block {block_idx} "
                  f"(of {self.num_cnn_blocks} total blocks)")

        # Always cache CNN final output for efficient policy target computation
        def cnn_final_hook(module, input, output):
            self._cached_cnn_final = output

        self._hooks.append(
            self.maia.maia.chess_cnn.register_forward_hook(cnn_final_hook)
        )

        # Hook on mid-transformer layer (optional)
        if self.use_transformer_taps:
            transformer = self.maia.maia.transformer
            if hasattr(transformer, 'elo_layers') and len(transformer.elo_layers) > 0:
                num_layers = len(transformer.elo_layers)
                mid_idx = 0
                ff_module = transformer.elo_layers[mid_idx][1]

                def mid_hook(module, input, output):
                    if isinstance(output, tuple):
                        self._mid_output = output[0]
                    else:
                        self._mid_output = output

                self._hooks.append(
                    ff_module.register_forward_hook(mid_hook)
                )
                print(f"  [MultiScale] Hooked mid-transformer at elo_layers[{mid_idx}].ff ({num_layers} total layers)")
            else:
                print("  [MultiScale] WARNING: Could not find elo_layers, mid-layer hook skipped")

    def _build_spatial_pos_enc(self, batch_size: int,
                                device: torch.device) -> torch.Tensor:
        """
        Build per-square absolute positional encoding.

        The 8x8 grid uses fixed rank (0-7) x file (0-7) indices corresponding
        to absolute board coordinates (rank 0 = rank 1, file 0 = a-file).
        This is intentionally NOT side-conditioned so the LLM can reference
        squares like "d4" directly through the learned embeddings.

        Returns: (B, 64, 16) positional encoding per square
        """
        rank_idx = torch.arange(8, device=device).repeat_interleave(8)  # (64,)
        file_idx = torch.arange(8, device=device).repeat(8)              # (64,)

        rank_emb = self.rank_embedding(rank_idx)  # (64, 8)
        file_emb = self.file_embedding(file_idx)  # (64, 8)

        pos_enc = torch.cat([rank_emb, file_emb], dim=-1)  # (64, 16)
        return pos_enc.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 64, 16)

    def forward(
        self,
        boards: torch.Tensor,
        elo_self: torch.Tensor,
        elo_oppo: torch.Tensor,
        side_to_move: torch.Tensor = None,
        abs_boards: torch.Tensor = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Extract multi-scale features and concatenate.

        Args:
            boards: (B, 18, 8, 8) â€” perspective-relative (mirrored for Black)
            elo_self: (B,) or None
            elo_oppo: (B,) or None
            side_to_move: (B,) long tensor, 1=White, 0=Black
            abs_boards: (B, 18, 8, 8) â€” absolute coordinates (un-mirrored).
                        When provided, CNN features are spatially un-mirrored
                        and abs_boards is used for CSMP piece/mask computation.

        Returns:
            (B, S, tap_dim) where S = N_cnn_taps*64 + [16 if transformer] + 1
        """
        batch_size = boards.size(0)
        device = boards.device

        context_parts = []
        csmp_square_tokens = None

        if self.use_cnn:
            # Run backbone to populate hooks (backbone expects perspective-relative boards)
            if self.use_transformer_taps:
                final_seq = self.maia.get_transformer_output(
                    boards, elo_self, elo_oppo, return_sequence=True
                )
            else:
                self.maia.get_cnn_output(boards)

            # Collect raw CNN tap outputs; un-mirror spatially if abs_boards provided
            raw_taps = [self._cnn_outputs[idx] for idx in self.cnn_tap_layers]
            if abs_boards is not None and side_to_move is not None:
                raw_taps = _unmirror_cnn_features(raw_taps, side_to_move)
        else:
            raw_taps = []  # CNN-free: no CNN features
            final_seq = None

        # Board tensor for CSMP / piece embeddings / dynamic masks:
        # use absolute-space board when available
        csmp_boards = abs_boards if abs_boards is not None else boards

        # --- CNN taps (or CNN-free CSMP) ---
        if self.use_csmp:
            # CSMP path: handles both CNN and CNN-free (empty raw_taps)
            l_cnn = self.chess_mp(raw_taps, csmp_boards)  # (B, 64, tap_dim)
            csmp_square_tokens = l_cnn
            context_parts.append(l_cnn)
        elif self.concat_cnn_taps:
            # Legacy concat mode: stack all tap outputs per square, then single projection
            pos_enc = self._build_spatial_pos_enc(batch_size, device)  # (B, 64, 16)
            tap_spatials = []
            for cnn_out in raw_taps:
                cnn_spatial = cnn_out.permute(0, 2, 3, 1).reshape(batch_size, 64, self.cnn_dim)
                tap_spatials.append(cnn_spatial)
            # (B, 64, N_taps * 256)
            multi_scale = torch.cat(tap_spatials, dim=-1)
            multi_scale_with_pos = torch.cat(
                [multi_scale, pos_enc.to(multi_scale.dtype)], dim=-1
            )  # (B, 64, N_taps*256 + 16)
            l_cnn = self.proj_cnn_concat(multi_scale_with_pos)  # (B, 64, tap_dim)
            context_parts.append(l_cnn)
        else:
            # Legacy per-tap mode: separate tokens per tap layer
            pos_enc = self._build_spatial_pos_enc(batch_size, device)  # (B, 64, 16)
            for tap_i, cnn_out in enumerate(raw_taps):
                cnn_spatial = cnn_out.permute(0, 2, 3, 1).reshape(batch_size, 64, self.cnn_dim)
                cnn_with_pos = torch.cat([cnn_spatial, pos_enc.to(cnn_spatial.dtype)], dim=-1)  # (B, 64, 272)
                l_cnn = self.proj_cnn[tap_i](cnn_with_pos)  # (B, 64, tap_dim)
                context_parts.append(l_cnn)

        # --- Transformer taps (optional): channel-tokens in native shape ---
        if self.use_transformer_taps:
            if self._mid_output is not None:
                l_mid = self.proj_mid(self._mid_output)
            else:
                l_mid = self.proj_mid(final_seq)
            context_parts.append(l_mid)

            l_final = self.proj_final(final_seq)
            context_parts.append(l_final)

        # --- Side-to-move token ---
        if side_to_move is not None:
            if isinstance(side_to_move, torch.Tensor):
                side_idx = side_to_move.long().to(device)
            else:
                side_idx = torch.tensor(
                    [1 if s else 0 for s in side_to_move],
                    dtype=torch.long, device=device
                )
        else:
            side_idx = torch.ones(batch_size, dtype=torch.long, device=device)
        # Cast to match CNN tap dtype (bf16 compat)
        ref_dtype = context_parts[0].dtype if context_parts else torch.float32
        side_tok = self.side_token(side_idx).unsqueeze(1).to(ref_dtype)  # (B, 1, tap_dim)
        context_parts.append(side_tok)

        context = torch.cat(context_parts, dim=1)

        if return_components:
            return {
                "context": context,
                "csmp_square_tokens": csmp_square_tokens,
            }

        return context


# =============================================================================
# Square Latent Encoder
# =============================================================================

class SquareLatentEncoder(nn.Module):
    """
    Perceiver that processes multi-scale chess context with learned latents.

    Context size depends on configuration:
      - Without transformer taps: (B, 65, tap_dim)   â€” 64 spatial CNN + 1 side token
      - With transformer taps:    (B, 81, tap_dim)   â€” 64 spatial CNN + 8 mid + 8 final + 1 side token

    Produces:
      - latents: (B, num_latents, perceiver_dim) â€” shared latents for per-layer xattn readout
      - aux_repr via 1-query attention pool  -> (B, perceiver_dim) for auxiliary heads
      - policy_logits                        -> (B, 1880)
      - eval_logits (optional)               -> (B, num_eval_buckets)
    """

    def __init__(
        self,
        tap_dim: int = 1024,
        perceiver_dim: int = 2048,
        num_latents: int = 64,
        depth: int = 4,
        heads: int = 16,
        num_eval_buckets: int = 5,
        enable_eval_head: bool = True,
        use_engineered_concat: bool = False,
        engineered_dim: int = 204,
        dropout: float = 0.1,
        structured_latents: bool = False,
        latent_context_mask_type: str = "full",
        global_latent_attends_all: bool = True,
        square_latent_attends_side_token: bool = True,
        use_structured_policy_head: bool = False,
        policy_include_global_latent: bool = True,
        structured_policy_query_layers: int = 4,
        structured_policy_query_heads: Optional[int] = None,
        structured_policy_ffn_mult: int = 2,
        structured_policy_use_move_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.tap_dim = tap_dim
        self.perceiver_dim = perceiver_dim
        self.num_latents = num_latents
        self.depth = depth
        self.use_engineered_concat = use_engineered_concat
        self.engineered_dim = engineered_dim
        self.structured_latents = bool(structured_latents)
        self.latent_context_mask_type = str(latent_context_mask_type)
        self.global_latent_attends_all = bool(global_latent_attends_all)
        self.square_latent_attends_side_token = bool(square_latent_attends_side_token)
        self.use_structured_policy_head = bool(use_structured_policy_head)
        self.policy_include_global_latent = bool(policy_include_global_latent)
        self.structured_policy_query_layers = int(structured_policy_query_layers)
        self.structured_policy_query_heads = (
            int(structured_policy_query_heads)
            if structured_policy_query_heads is not None
            else int(heads)
        )
        self.structured_policy_ffn_mult = int(structured_policy_ffn_mult)
        self.structured_policy_use_move_bias = bool(structured_policy_use_move_bias)
        self.enable_eval_head = bool(enable_eval_head)
        self.drop = nn.Dropout(dropout)

        if self.structured_latents and num_latents != 65:
            raise ValueError(
                f"structured_latents=True requires num_latents=65 (got {num_latents})"
            )
        if self.structured_latents and self.latent_context_mask_type not in {"full", "strict_own_square"}:
            raise ValueError(
                "latent_context_mask_type must be one of {'full', 'strict_own_square'} "
                f"(got {self.latent_context_mask_type!r})"
            )
        if self.structured_latents and use_engineered_concat:
            raise ValueError("structured_latents mode does not support use_engineered_concat=True")
        if self.use_structured_policy_head:
            if self.structured_policy_query_layers <= 0:
                raise ValueError("structured_policy_query_layers must be > 0")
            if self.structured_policy_query_heads <= 0:
                raise ValueError("structured_policy_query_heads must be > 0")
            if perceiver_dim % self.structured_policy_query_heads != 0:
                raise ValueError(
                    "structured_policy_query_heads must divide perceiver_dim "
                    f"(got heads={self.structured_policy_query_heads}, perceiver_dim={perceiver_dim})"
                )
            if self.structured_policy_ffn_mult <= 0:
                raise ValueError("structured_policy_ffn_mult must be > 0")

        # Latent base dimension (reduced if engineered features are concatenated)
        if use_engineered_concat:
            self.latent_base_dim = perceiver_dim - engineered_dim
        else:
            self.latent_base_dim = perceiver_dim

        # Latent queries:
        # - structured_latents=True: shared square base + separate global base (64+1 slots)
        # - structured_latents=False: independent learned latent bank (legacy behavior)
        if self.structured_latents:
            self.square_latent_base = nn.Parameter(torch.randn(1, 1, self.latent_base_dim) * 0.02)
            self.global_latent_base = nn.Parameter(torch.randn(1, 1, self.latent_base_dim) * 0.02)
            self.register_parameter("latents", None)
        else:
            self.latents = nn.Parameter(torch.randn(1, num_latents, self.latent_base_dim) * 0.02)
            self.register_parameter("square_latent_base", None)
            self.register_parameter("global_latent_base", None)

        if self.use_structured_policy_head:
            if get_maia_mapping is None:
                raise ImportError("Structured policy head requires maia2 move mapping (training.maia_model.get_maia_mapping)")
            mapping = get_maia_mapping()
            from_sq_idx: List[int] = []
            to_sq_idx: List[int] = []
            for move_uci in mapping.vocab:
                if len(move_uci) < 4:
                    raise ValueError(f"Unexpected Maia move encoding: {move_uci!r}")
                from_sq = self._uci_square_to_index(move_uci[:2])
                to_sq = self._uci_square_to_index(move_uci[2:4])
                if from_sq < 0 or to_sq < 0:
                    raise ValueError(f"Could not parse Maia move: {move_uci!r}")
                from_sq_idx.append(from_sq)
                to_sq_idx.append(to_sq)

            self.register_buffer("policy_from_square_idx", torch.tensor(from_sq_idx, dtype=torch.long))
            self.register_buffer("policy_to_square_idx", torch.tensor(to_sq_idx, dtype=torch.long))

            # Structured policy readout:
            # - structured_latents=True: shared square-query base expanded to 64
            # - structured_latents=False: independent learned square queries
            if self.structured_latents:
                self.policy_square_query_base = nn.Parameter(torch.randn(1, 1, perceiver_dim) * 0.02)
                self.register_parameter("policy_square_queries", None)
            else:
                self.policy_square_queries = nn.Parameter(torch.randn(1, 64, perceiver_dim) * 0.02)
                self.register_parameter("policy_square_query_base", None)
            self.policy_cross_norm_q = nn.ModuleList()
            self.policy_cross_norm_kv = nn.ModuleList()
            self.policy_square_k_proj = nn.ModuleList()
            self.policy_square_v_proj = nn.ModuleList()
            self.policy_global_k_proj = nn.ModuleList()
            self.policy_global_v_proj = nn.ModuleList()
            self.policy_self_norm = nn.ModuleList()
            self.policy_self_attn = nn.ModuleList()
            self.policy_ffn_norm = nn.ModuleList()
            self.policy_ffn = nn.ModuleList()

            policy_ffn_hidden = perceiver_dim * self.structured_policy_ffn_mult
            for _ in range(self.structured_policy_query_layers):
                self.policy_cross_norm_q.append(nn.LayerNorm(perceiver_dim))
                self.policy_cross_norm_kv.append(nn.LayerNorm(perceiver_dim))
                self.policy_square_k_proj.append(nn.Linear(perceiver_dim, perceiver_dim))
                self.policy_square_v_proj.append(nn.Linear(perceiver_dim, perceiver_dim))
                # Separate K/V projections for the global latent token.
                self.policy_global_k_proj.append(nn.Linear(perceiver_dim, perceiver_dim))
                self.policy_global_v_proj.append(nn.Linear(perceiver_dim, perceiver_dim))

                self.policy_self_norm.append(nn.LayerNorm(perceiver_dim))
                self.policy_self_attn.append(ManualMultiHeadAttention(
                    perceiver_dim,
                    self.structured_policy_query_heads,
                    dropout=dropout,
                ))

                self.policy_ffn_norm.append(nn.LayerNorm(perceiver_dim))
                self.policy_ffn.append(nn.Sequential(
                    nn.Linear(perceiver_dim, policy_ffn_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(policy_ffn_hidden, perceiver_dim),
                    nn.Dropout(dropout),
                ))

            self.policy_from_proj = nn.Linear(perceiver_dim, perceiver_dim)
            self.policy_to_proj = nn.Linear(perceiver_dim, perceiver_dim)
            if self.structured_policy_use_move_bias:
                self.policy_move_bias = nn.Parameter(torch.zeros(len(from_sq_idx)))
            else:
                self.register_parameter("policy_move_bias", None)

            # Shared policy latents are passed to LLM fusion; each prediction
            # head then gets its own final structured refinement layer.
            self.policy_distill_branch = StructuredSquareBranchLayer(
                perceiver_dim=perceiver_dim,
                heads=self.structured_policy_query_heads,
                ffn_mult=self.structured_policy_ffn_mult,
                dropout=dropout,
                include_global_latent=self.policy_include_global_latent,
            )
            self.eval_ce_branch = StructuredSquareBranchLayer(
                perceiver_dim=perceiver_dim,
                heads=self.structured_policy_query_heads,
                ffn_mult=self.structured_policy_ffn_mult,
                dropout=dropout,
                include_global_latent=self.policy_include_global_latent,
            )
            self.eval_mse_branch = StructuredSquareBranchLayer(
                perceiver_dim=perceiver_dim,
                heads=self.structured_policy_query_heads,
                ffn_mult=self.structured_policy_ffn_mult,
                dropout=dropout,
                include_global_latent=self.policy_include_global_latent,
            )

        # Perceiver layers: self-attn -> cross-attn -> FFN
        self.self_attn_layers = nn.ModuleList()
        self.self_attn_norms = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.cross_attn_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(depth):
            # Self-attention among latents
            self.self_attn_layers.append(
                ManualMultiHeadAttention(perceiver_dim, heads, dropout=dropout)
            )
            self.self_attn_norms.append(nn.LayerNorm(perceiver_dim))

            # Cross-attention: latents query the multi-scale context
            self.cross_attn_layers.append(
                ManualMultiHeadAttention(
                    perceiver_dim, heads,
                    kdim=tap_dim, vdim=tap_dim,
                    dropout=dropout,
                )
            )
            self.cross_attn_norms.append(nn.LayerNorm(perceiver_dim))

            # FFN
            self.ffns.append(nn.Sequential(
                nn.Linear(perceiver_dim, perceiver_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(perceiver_dim * 4, perceiver_dim),
                nn.Dropout(dropout),
            ))
            self.ffn_norms.append(nn.LayerNorm(perceiver_dim))

        # Final LayerNorm
        self.ln_final = nn.LayerNorm(perceiver_dim)

        # --- Dual Attention Pool Branches ---

        # Branch A: Auxiliary (1 learned query -> policy/eval heads)
        self.aux_query = nn.Parameter(torch.randn(1, 1, perceiver_dim) * 0.02)
        self.aux_pool_attn = ManualMultiHeadAttention(
            perceiver_dim, heads,
        )
        self.aux_pool_norm = nn.LayerNorm(perceiver_dim)

        # Auxiliary heads
        self.policy_head = nn.Linear(perceiver_dim, 1880)
        if self.enable_eval_head:
            self.eval_head = nn.Linear(perceiver_dim, num_eval_buckets)
        else:
            self.eval_head = None

        # Per-move evaluation head (structured, shares square readout with policy head)
        move_eval_dim = int(kwargs.get("move_eval_dim", 128))
        self.move_eval_dim = move_eval_dim
        if self.use_structured_policy_head:
            self.eval_from_proj = nn.Linear(perceiver_dim, move_eval_dim)
            self.eval_to_proj = nn.Linear(perceiver_dim, move_eval_dim)
            self.eval_move_bias = nn.Parameter(torch.zeros(len(from_sq_idx) if hasattr(self, 'policy_from_square_idx') else 1880))
            self.eval_mse_from_proj = nn.Linear(perceiver_dim, move_eval_dim)
            self.eval_mse_to_proj = nn.Linear(perceiver_dim, move_eval_dim)
            self.eval_mse_move_bias = nn.Parameter(torch.zeros(len(from_sq_idx) if hasattr(self, 'policy_from_square_idx') else 1880))
            self.mate_from_proj = nn.Linear(perceiver_dim, move_eval_dim)
            self.mate_to_proj = nn.Linear(perceiver_dim, move_eval_dim)
            self.mate_move_bias = nn.Parameter(torch.zeros(len(from_sq_idx) if hasattr(self, 'policy_from_square_idx') else 1880))
        else:
            self.eval_from_proj = None
            self.eval_to_proj = None
            self.eval_mse_from_proj = None
            self.eval_mse_to_proj = None
            self.mate_from_proj = None
            self.mate_to_proj = None

        # Entropy logging flag â€” set to True by training loop near logging steps
        self.log_entropy = False

        print(f"  SquareLatentEncoder: {num_latents} latents, {depth} layers, dim={perceiver_dim}")
        if self.structured_latents:
            print(
                "    Structured latents enabled: 64 shared-base square slots + 1 global slot, "
                f"mask={self.latent_context_mask_type}, square->side={self.square_latent_attends_side_token}, "
                f"global->all={self.global_latent_attends_all}"
            )
        if self.enable_eval_head:
            print(f"    Aux pool: 1 query -> policy(1880) + eval({num_eval_buckets})")
        else:
            print("    Aux pool: 1 query -> policy(1880) (eval bucket head disabled)")
        if self.use_structured_policy_head:
            print(
                "    Policy core: square-query cross/self-attn readout "
                f"(layers={self.structured_policy_query_layers}, heads={self.structured_policy_query_heads}), "
                "shared with LLM policy_latents"
            )
            print("    Prediction branches: 3 x (structured x-attn + full self-attn + FFN)")
            print("      - Maia policy distillation")
            print("      - Eval CE + pairwise")
            print(f"      - Eval MSE + mate (dim={move_eval_dim})")
        print(f"    Latents passed to per-layer xattn readout heads (no fusion pool)")

    @staticmethod
    def _uci_square_to_index(square_uci: str) -> int:
        if len(square_uci) != 2:
            return -1
        file_char, rank_char = square_uci[0], square_uci[1]
        if file_char < 'a' or file_char > 'h' or rank_char < '1' or rank_char > '8':
            return -1
        file_idx = ord(file_char) - ord('a')
        rank_idx = int(rank_char) - 1
        return rank_idx * 8 + file_idx

    @staticmethod
    def _mirror_square_indices(square_idx: torch.Tensor) -> torch.Tensor:
        rank = torch.div(square_idx, 8, rounding_mode='floor')
        file = square_idx % 8
        return (7 - rank) * 8 + file

    def _structured_policy_square_readout(self, latents: torch.Tensor) -> torch.Tensor:
        """Refine 64 square policy representations from Perceiver latents."""
        if latents.size(1) < 64:
            raise ValueError(
                f"Structured policy square readout requires >=64 latents, got {latents.size(1)}"
            )

        batch_size = latents.size(0)
        if self.structured_latents and self.policy_square_query_base is not None:
            x = self.policy_square_query_base.expand(batch_size, 64, -1)  # (B, 64, D)
        else:
            x = self.policy_square_queries.expand(batch_size, -1, -1)  # (B, 64, D)
        attn_scale = 1.0 / math.sqrt(float(self.perceiver_dim))

        for i in range(self.structured_policy_query_layers):
            q = self.policy_cross_norm_q[i](x)
            kv = self.policy_cross_norm_kv[i](latents)

            sq_src = kv[:, :64, :]
            sq_k = self.policy_square_k_proj[i](sq_src)
            sq_v = self.policy_square_v_proj[i](sq_src)

            has_global = bool(self.policy_include_global_latent and latents.size(1) > 64)
            if has_global:
                gl_src = kv[:, 64:65, :]
                gl_k = self.policy_global_k_proj[i](gl_src)
                gl_v = self.policy_global_v_proj[i](gl_src)
                gl_logits = torch.matmul(q, gl_k.transpose(-2, -1)) * attn_scale  # (B, 64, 1)
                gl_v_expanded = gl_v.expand(-1, 64, -1)
            else:
                gl_logits = torch.full(
                    (batch_size, 64, 1),
                    fill_value=-1e4,
                    dtype=q.dtype,
                    device=q.device,
                )
                gl_v_expanded = torch.zeros_like(sq_v)

            sq_logits = (q * sq_k).sum(dim=-1, keepdim=True) * attn_scale  # (B, 64, 1)
            mix = torch.softmax(torch.cat([sq_logits, gl_logits], dim=-1), dim=-1)  # (B, 64, 2)
            ca_out = mix[..., :1] * sq_v + mix[..., 1:] * gl_v_expanded
            x = x + self.drop(ca_out)

            x_norm = self.policy_self_norm[i](x)
            sa_out, _ = self.policy_self_attn[i](
                query=x_norm,
                key=x_norm,
                value=x_norm,
                need_weights=False,
            )
            x = x + self.drop(sa_out)

            x_norm = self.policy_ffn_norm[i](x)
            x = x + self.policy_ffn[i](x_norm)

        return x

    def _build_latent_context_attn_mask(self, context_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.structured_latents:
            return None
        if self.latent_context_mask_type == "full":
            return None

        # Mask convention: True = masked-out (ManualMultiHeadAttention inverts internally)
        mask = torch.ones(self.num_latents, context_len, dtype=torch.bool, device=device)

        n_sq = min(64, context_len, self.num_latents)
        if n_sq > 0:
            sq_idx = torch.arange(n_sq, device=device)
            mask[sq_idx, sq_idx] = False

        if self.square_latent_attends_side_token and context_len > 0 and self.num_latents >= 64:
            mask[:64, context_len - 1] = False

        if self.num_latents > 64:
            if self.global_latent_attends_all:
                mask[64, :] = False
            elif context_len > 0:
                mask[64, context_len - 1] = False

        return mask

    def _build_policy_branch_source_latents(
        self,
        policy_latents: torch.Tensor,
        perceiver_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Use core policy latents as branch KV, with optional Perceiver global latent."""
        if not self.policy_include_global_latent or perceiver_latents.size(1) <= 64:
            return policy_latents
        return torch.cat([policy_latents, perceiver_latents[:, 64:65, :]], dim=1)

    def _compute_move_head_logits(
        self,
        square_repr: torch.Tensor,
        from_proj: Optional[nn.Linear],
        to_proj: Optional[nn.Linear],
        move_bias: Optional[torch.Tensor],
        side_to_move: Optional[torch.Tensor] = None,
        latents_are_absolute: bool = True,
    ) -> Optional[torch.Tensor]:
        """Shared from/to move scorer for policy, ranking, regression, and mate heads."""
        if from_proj is None or to_proj is None:
            return None

        batch_size = square_repr.size(0)
        from_sq_idx = self.policy_from_square_idx.unsqueeze(0).expand(batch_size, -1)
        to_sq_idx = self.policy_to_square_idx.unsqueeze(0).expand(batch_size, -1)

        if latents_are_absolute:
            if side_to_move is None:
                side_tensor = torch.ones(batch_size, dtype=torch.long, device=square_repr.device)
            else:
                side_tensor = side_to_move.to(device=square_repr.device, dtype=torch.long).view(-1)
                if side_tensor.numel() != batch_size:
                    raise ValueError(
                        f"side_to_move batch mismatch: expected {batch_size}, got {side_tensor.numel()}"
                    )

            black_mask = (side_tensor == 0).unsqueeze(1)
            if black_mask.any():
                mirrored_from = self._mirror_square_indices(self.policy_from_square_idx)
                mirrored_to = self._mirror_square_indices(self.policy_to_square_idx)
                from_sq_idx = torch.where(black_mask, mirrored_from.unsqueeze(0), from_sq_idx)
                to_sq_idx = torch.where(black_mask, mirrored_to.unsqueeze(0), to_sq_idx)

        from_space = from_proj(square_repr)
        to_space = to_proj(square_repr)
        d_model = from_space.size(-1)

        from_repr = from_space.gather(
            dim=1,
            index=from_sq_idx.unsqueeze(-1).expand(-1, -1, d_model),
        )
        to_repr = to_space.gather(
            dim=1,
            index=to_sq_idx.unsqueeze(-1).expand(-1, -1, d_model),
        )
        logits = (from_repr * to_repr).sum(dim=-1) / math.sqrt(float(d_model))
        if move_bias is not None:
            logits = logits + move_bias.unsqueeze(0)
        return logits

    def _compute_structured_policy_logits(
        self,
        latents: torch.Tensor,
        side_to_move: Optional[torch.Tensor] = None,
        latents_are_absolute: bool = True,
        square_repr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents.size(1) < 64:
            raise ValueError(
                f"Structured policy head requires >=64 latents, got {latents.size(1)}"
            )

        if square_repr is None:
            square_repr = self._structured_policy_square_readout(latents)
        return self._compute_move_head_logits(
            square_repr=square_repr,
            from_proj=self.policy_from_proj,
            to_proj=self.policy_to_proj,
            move_bias=self.policy_move_bias,
            side_to_move=side_to_move,
            latents_are_absolute=latents_are_absolute,
        )

    def _compute_move_eval_logits(
        self,
        square_repr: torch.Tensor,
        side_to_move: Optional[torch.Tensor] = None,
        latents_are_absolute: bool = True,
    ) -> torch.Tensor:
        """Compute per-move evaluation scores using from/to eval projections.

        Reuses the square_repr from the policy square readout (shared computation).

        Args:
            square_repr: (B, 64, perceiver_dim) from _structured_policy_square_readout
            side_to_move: (B,) 1=White, 0=Black
            latents_are_absolute: whether square indexing is absolute

        Returns:
            (B, 1880) raw eval scores per move (pre-bucket-projection)
        """
        return self._compute_move_head_logits(
            square_repr=square_repr,
            from_proj=self.eval_from_proj,
            to_proj=self.eval_to_proj,
            move_bias=self.eval_move_bias,
            side_to_move=side_to_move,
            latents_are_absolute=latents_are_absolute,
        )

    def _compute_move_eval_mse_logits(
        self,
        square_repr: torch.Tensor,
        side_to_move: Optional[torch.Tensor] = None,
        latents_are_absolute: bool = True,
    ) -> Optional[torch.Tensor]:
        """Compute per-move regression scores on the MSE-specialized branch."""
        return self._compute_move_head_logits(
            square_repr=square_repr,
            from_proj=self.eval_mse_from_proj,
            to_proj=self.eval_mse_to_proj,
            move_bias=self.eval_mse_move_bias,
            side_to_move=side_to_move,
            latents_are_absolute=latents_are_absolute,
        )

    def _compute_move_mate_logits(
        self,
        square_repr: torch.Tensor,
        side_to_move: Optional[torch.Tensor] = None,
        latents_are_absolute: bool = True,
    ) -> torch.Tensor:
        """Compute per-move mate logits using dedicated from/to mate projections."""
        return self._compute_move_head_logits(
            square_repr=square_repr,
            from_proj=self.mate_from_proj,
            to_proj=self.mate_to_proj,
            move_bias=self.mate_move_bias,
            side_to_move=side_to_move,
            latents_are_absolute=latents_are_absolute,
        )

    @staticmethod
    def _attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute mean entropy of attention distributions.

        Args:
            attn_weights: (B, H, Q, K) â€” softmax attention weights

        Returns:
            (H,) â€” mean entropy per head (averaged over batch and queries)
        """
        eps = 1e-8
        # H = -sum(p * log(p))
        log_p = torch.log(attn_weights + eps)
        entropy = -(attn_weights * log_p).sum(dim=-1)  # (B, H, Q)
        return entropy.mean(dim=(0, 2))  # (H,)

    def forward(
        self,
        context: torch.Tensor,
        engineered_features: Optional[torch.Tensor] = None,
        side_to_move: Optional[torch.Tensor] = None,
        latents_are_absolute: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Args:
            context: (B, S, tap_dim) multi-scale features (S=65 or 81)
                     Side-to-move is already encoded as a context token.
            engineered_features: (B, 64, engineered_dim) if use_engineered_concat

        Returns:
            latents: (B, num_latents, perceiver_dim) â€” for per-layer xattn readout
            aux_repr: (B, perceiver_dim)
            policy_logits: (B, 1880)
            eval_logits: (B, num_eval_buckets) or None
            policy_latents: (B, 64, perceiver_dim) or None
            entropy_metrics: dict with per-layer and mean cross-attention entropy
            move_eval_logits: (B, 1880) eval CE/pairwise branch logits or None
            move_eval_mse_logits: (B, 1880) eval MSE branch logits or None
            move_mate_logits: (B, 1880) or None
        """
        batch_size = context.size(0)

        # Expand latents
        if self.structured_latents:
            sq = self.square_latent_base.expand(batch_size, 64, -1)
            gl = self.global_latent_base.expand(batch_size, 1, -1)
            x = torch.cat([sq, gl], dim=1)  # (B, 65, latent_base_dim)
        else:
            x = self.latents.expand(batch_size, -1, -1)

        # Concatenate engineered features if enabled
        if self.use_engineered_concat:
            if engineered_features is None:
                raise ValueError("engineered_features required when use_engineered_concat=True")
            engineered_features = engineered_features.to(x.device)
            x = torch.cat([x, engineered_features], dim=-1)  # (B, N, perceiver_dim)

        # Perceiver layers (Pre-LN)
        layer_entropies = []
        square_own_masses = []
        square_side_masses = []
        global_latent_spatial_masses = []
        compute_entropy = self.log_entropy
        cross_attn_mask = self._build_latent_context_attn_mask(context.size(1), context.device)
        for i in range(self.depth):
            # 1. Self-Attention
            x_norm = self.self_attn_norms[i](x)
            sa_out, _ = self.self_attn_layers[i](query=x_norm, key=x_norm, value=x_norm, need_weights=False)
            x = x + self.drop(sa_out)

            # 2. Cross-Attention to multi-scale context
            x_norm = self.cross_attn_norms[i](x)
            if compute_entropy and not _compiler_is_compiling():
                # Materialize per-head weights for entropy logging
                # (manual softmax path â€” not traced by torch.compile)
                ca_out, ca_weights = self.cross_attn_layers[i](
                    query=x_norm, key=context, value=context,
                    need_weights=True, average_attn_weights=False,
                    attn_mask=cross_attn_mask,
                )  # ca_weights: (B, H, num_latents, S)
                layer_entropies.append(self._attention_entropy(ca_weights.detach()))

                if self.structured_latents:
                    n_sq = min(64, ca_weights.size(-2), ca_weights.size(-1))
                    if n_sq > 0:
                        sq_weights = ca_weights[:, :, :n_sq, :]  # (B, H, n_sq, S)
                        own_idx = torch.arange(n_sq, device=ca_weights.device).view(1, 1, n_sq, 1)
                        own_idx = own_idx.expand(ca_weights.size(0), ca_weights.size(1), n_sq, 1)
                        own_mass = sq_weights.gather(-1, own_idx).mean()
                        square_own_masses.append(own_mass.detach())

                        side_idx = ca_weights.size(-1) - 1
                        side_mass = sq_weights[..., side_idx].mean()
                        square_side_masses.append(side_mass.detach())

                    if ca_weights.size(-2) > 64:
                        global_mass = ca_weights[:, :, 64, :min(64, ca_weights.size(-1))].mean()
                        global_latent_spatial_masses.append(global_mass.detach())
            else:
                # Fast path â€” SDPA kernel, no weight materialization
                ca_out, _ = self.cross_attn_layers[i](
                    query=x_norm, key=context, value=context,
                    attn_mask=cross_attn_mask,
                    need_weights=False,
                )
            x = x + self.drop(ca_out)

            # 3. FFN
            x_norm = self.ffn_norms[i](x)
            x = x + self.ffns[i](x_norm)

        # Final LayerNorm
        x = self.ln_final(x)  # (B, num_latents, perceiver_dim)

        # --- Branch A: Auxiliary ---
        aux_q = self.aux_query.expand(batch_size, -1, -1)  # (B, 1, D)
        aux_q_norm = self.aux_pool_norm(aux_q)
        aux_repr, _ = self.aux_pool_attn(query=aux_q_norm, key=x, value=x, need_weights=False)
        aux_repr = aux_repr.squeeze(1)  # (B, D)

        move_eval_logits = None
        move_eval_mse_logits = None
        move_mate_logits = None
        policy_latents = None
        policy_head_ms = None
        if self.use_structured_policy_head:
            _ph_t0 = None
            if getattr(self, "_profile", False):
                import time
                torch.cuda.synchronize(x.device)
                _ph_t0 = time.perf_counter()
            policy_core_latents = self._structured_policy_square_readout(x)
            policy_latents = policy_core_latents
            branch_source_latents = self._build_policy_branch_source_latents(
                policy_core_latents, x,
            )
            policy_branch_latents = self.policy_distill_branch(policy_core_latents, branch_source_latents)
            policy_logits = self._compute_structured_policy_logits(
                x,
                side_to_move=side_to_move,
                latents_are_absolute=latents_are_absolute,
                square_repr=policy_branch_latents,
            )
            # Compute specialized move-eval branches on top of shared policy latents.
            if self.eval_from_proj is not None:
                eval_ce_latents = self.eval_ce_branch(policy_core_latents, branch_source_latents)
                eval_mse_latents = self.eval_mse_branch(policy_core_latents, branch_source_latents)
                move_eval_logits = self._compute_move_eval_logits(
                    eval_ce_latents, side_to_move=side_to_move,
                    latents_are_absolute=latents_are_absolute,
                )
                move_eval_mse_logits = self._compute_move_eval_mse_logits(
                    eval_mse_latents, side_to_move=side_to_move,
                    latents_are_absolute=latents_are_absolute,
                )
                move_mate_logits = self._compute_move_mate_logits(
                    eval_mse_latents, side_to_move=side_to_move,
                    latents_are_absolute=latents_are_absolute,
                )
            if _ph_t0 is not None:
                torch.cuda.synchronize(x.device)
                policy_head_ms = (time.perf_counter() - _ph_t0) * 1000.0
        else:
            policy_logits = self.policy_head(aux_repr)  # (B, 1880)
        eval_logits = self.eval_head(aux_repr) if self.eval_head is not None else None

        # Entropy metrics (detached, for logging only)
        entropy_metrics = {}
        if layer_entropies:
            stacked = torch.stack(layer_entropies, dim=0)  # (depth, H)
            entropy_metrics['cross_attn_entropy_mean'] = stacked.mean()
            entropy_metrics['cross_attn_entropy_per_layer'] = stacked.mean(dim=1)  # (depth,)
            entropy_metrics['cross_attn_entropy_per_head'] = stacked.mean(dim=0)   # (H,)
        if square_own_masses:
            entropy_metrics['cross_attn_square_to_own_mass'] = torch.stack(square_own_masses).mean()
        if square_side_masses:
            entropy_metrics['cross_attn_square_to_side_mass'] = torch.stack(square_side_masses).mean()
        if global_latent_spatial_masses:
            entropy_metrics['cross_attn_global_to_spatial_mass'] = torch.stack(global_latent_spatial_masses).mean()
        if policy_head_ms is not None:
            entropy_metrics['policy_head_ms'] = torch.tensor(
                policy_head_ms, device=x.device, dtype=torch.float32
            )

        return (
            x,
            aux_repr,
            policy_logits,
            eval_logits,
            policy_latents,
            entropy_metrics,
            move_eval_logits,
            move_eval_mse_logits,
            move_mate_logits,
        )


# =============================================================================
# Structured Prediction Branch Layer
# =============================================================================

class StructuredSquareBranchLayer(nn.Module):
    """One structured x-attn + full self-attn refinement layer over 64 square latents."""

    def __init__(
        self,
        perceiver_dim: int,
        heads: int,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        include_global_latent: bool = True,
    ):
        super().__init__()
        self.perceiver_dim = int(perceiver_dim)
        self.include_global_latent = bool(include_global_latent)
        self.drop = nn.Dropout(dropout)

        self.cross_norm_q = nn.LayerNorm(perceiver_dim)
        self.cross_norm_kv = nn.LayerNorm(perceiver_dim)
        self.square_k_proj = nn.Linear(perceiver_dim, perceiver_dim)
        self.square_v_proj = nn.Linear(perceiver_dim, perceiver_dim)
        self.global_k_proj = nn.Linear(perceiver_dim, perceiver_dim)
        self.global_v_proj = nn.Linear(perceiver_dim, perceiver_dim)

        self.self_norm = nn.LayerNorm(perceiver_dim)
        self.self_attn = ManualMultiHeadAttention(
            perceiver_dim,
            heads,
            dropout=dropout,
        )

        ffn_hidden = perceiver_dim * int(ffn_mult)
        self.ffn_norm = nn.LayerNorm(perceiver_dim)
        self.ffn = nn.Sequential(
            nn.Linear(perceiver_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, perceiver_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, source_latents: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 64:
            raise ValueError(f"StructuredSquareBranchLayer expects 64 square tokens, got {x.size(1)}")
        if source_latents.size(1) < 64:
            raise ValueError(
                f"StructuredSquareBranchLayer requires >=64 source latents, got {source_latents.size(1)}"
            )

        q = self.cross_norm_q(x)
        kv = self.cross_norm_kv(source_latents)
        sq_src = kv[:, :64, :]
        sq_k = self.square_k_proj(sq_src)
        sq_v = self.square_v_proj(sq_src)
        attn_scale = 1.0 / math.sqrt(float(self.perceiver_dim))

        has_global = bool(self.include_global_latent and source_latents.size(1) > 64)
        if has_global:
            gl_src = kv[:, 64:65, :]
            gl_k = self.global_k_proj(gl_src)
            gl_v = self.global_v_proj(gl_src)
            gl_logits = torch.matmul(q, gl_k.transpose(-2, -1)) * attn_scale
            gl_v_expanded = gl_v.expand(-1, 64, -1)
        else:
            gl_logits = torch.full(
                (x.size(0), 64, 1),
                fill_value=-1e4,
                dtype=q.dtype,
                device=q.device,
            )
            gl_v_expanded = torch.zeros_like(sq_v)

        sq_logits = (q * sq_k).sum(dim=-1, keepdim=True) * attn_scale
        mix = torch.softmax(torch.cat([sq_logits, gl_logits], dim=-1), dim=-1)
        ca_out = mix[..., :1] * sq_v + mix[..., 1:] * gl_v_expanded
        x = x + self.drop(ca_out)

        x_norm = self.self_norm(x)
        sa_out, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        x = x + self.drop(sa_out)

        x_norm = self.ffn_norm(x)
        x = x + self.ffn(x_norm)
        return x


# =============================================================================
# Auxiliary Square Prediction Heads (BSR / SPP)
# =============================================================================

class AuxSquareHead(nn.Module):
    """
    Cross-attention readout head that attends to Perceiver latents and produces
    per-square predictions (64 squares).

    Architecture:
        64 learned queries (1, 64, head_dim)
        â†’ N layers of:
            Pre-LN cross-attention to Perceiver latents
            Pre-LN self-attention among 64 queries
            Pre-LN FFN (2x expansion)
        â†’ output_proj: head_dim â†’ output_dim per square
    """

    def __init__(
        self,
        perceiver_dim: int,
        head_dim: int = 256,
        output_dim: int = 13,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        structured_mode: bool = False,
        include_global_latent: bool = True,
        disable_query_self_attn_in_structured_mode: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.structured_mode = bool(structured_mode)
        self.include_global_latent = bool(include_global_latent)
        self.disable_query_self_attn_in_structured_mode = bool(disable_query_self_attn_in_structured_mode)

        # Square queries:
        # - structured_mode=True: shared query base expanded to 64
        # - structured_mode=False: independent learned square queries
        if self.structured_mode:
            self.square_query_base = nn.Parameter(torch.randn(1, 1, head_dim) * 0.02)
            self.register_parameter("queries", None)
        else:
            self.queries = nn.Parameter(torch.randn(1, 64, head_dim) * 0.02)
            self.register_parameter("square_query_base", None)

        # Per-layer modules
        self.cross_norm_q = nn.ModuleList()
        self.cross_norm_kv = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.self_norm = nn.ModuleList()
        self.self_attn = nn.ModuleList()
        self.ffn_norm = nn.ModuleList()
        self.ffn = nn.ModuleList()

        ffn_hidden = head_dim * 2
        for _ in range(n_layers):
            # Cross-attention: queries attend to Perceiver latents (Pre-LN)
            self.cross_norm_q.append(nn.LayerNorm(head_dim))
            self.cross_norm_kv.append(nn.LayerNorm(perceiver_dim))
            self.cross_attn.append(ManualMultiHeadAttention(
                head_dim, n_heads,
                kdim=perceiver_dim, vdim=perceiver_dim,
                dropout=dropout,
            ))

            # Self-attention among 64 square queries (Pre-LN)
            self.self_norm.append(nn.LayerNorm(head_dim))
            self.self_attn.append(ManualMultiHeadAttention(
                head_dim, n_heads, dropout=dropout,
            ))

            # FFN (Pre-LN)
            self.ffn_norm.append(nn.LayerNorm(head_dim))
            self.ffn.append(nn.Sequential(
                nn.Linear(head_dim, ffn_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden, head_dim),
            ))

        # Output projection
        self.output_proj = nn.Linear(head_dim, output_dim)

    def _build_structured_latent_mask(
        self,
        num_latents: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.structured_mode:
            return None
        if num_latents < 64:
            raise ValueError(f"AuxSquareHead structured_mode requires at least 64 latents (got {num_latents})")

        # MultiheadAttention convention: True = masked-out
        mask = torch.ones(64, num_latents, dtype=torch.bool, device=device)
        sq_idx = torch.arange(64, device=device)
        mask[sq_idx, sq_idx] = False  # each query attends to own square latent

        if self.include_global_latent and num_latents > 64:
            mask[:, 64] = False

        return mask

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: (B, num_latents, perceiver_dim) â€” Perceiver output

        Returns:
            (B, 64, output_dim) â€” per-square predictions
        """
        B = latents.size(0)
        if self.structured_mode and self.square_query_base is not None:
            x = self.square_query_base.expand(B, 64, -1)  # (B, 64, head_dim)
        else:
            x = self.queries.expand(B, -1, -1)  # (B, 64, head_dim)
        attn_mask = self._build_structured_latent_mask(latents.size(1), latents.device)

        for i in range(self.n_layers):
            # 1. Cross-attention to Perceiver latents
            q = self.cross_norm_q[i](x)
            kv = self.cross_norm_kv[i](latents)
            ca_out, _ = self.cross_attn[i](
                query=q,
                key=kv,
                value=kv,
                attn_mask=attn_mask,
                need_weights=False,
            )
            x = x + ca_out

            # 2. Self-attention among square queries
            if not (self.structured_mode and self.disable_query_self_attn_in_structured_mode):
                x_norm = self.self_norm[i](x)
                sa_out, _ = self.self_attn[i](query=x_norm, key=x_norm, value=x_norm, need_weights=False)
                x = x + sa_out

            # 3. FFN
            x_norm = self.ffn_norm[i](x)
            x = x + self.ffn[i](x_norm)

        return self.output_proj(x)  # (B, 64, output_dim)


# 8 ray directions: N, NE, E, SE, S, SW, W, NW
_RAY_DIRECTIONS = [
    (+1, 0), (+1, +1), (0, +1), (-1, +1),
    (-1, 0), (-1, -1), (0, -1), (+1, -1),
]

# Which directions each sliding piece type can use (indexed by piece type 0-12)
# Straights: indices 0,2,4,6 (N,E,S,W)  Diagonals: indices 1,3,5,7 (NE,SE,SW,NW)
_STRAIGHT_DIRS = {0, 2, 4, 6}
_DIAGONAL_DIRS = {1, 3, 5, 7}

# Piece types that use each ray set:
#   Rook (3,9): straights only
#   Bishop (2,8): diagonals only
#   Queen (4,10): all
# All others: no rays
_PIECE_RAY_MASK = {}  # piece_type -> set of allowed direction indices
for _pt in range(13):
    if _pt in (2, 8):       # Bishop
        _PIECE_RAY_MASK[_pt] = _DIAGONAL_DIRS
    elif _pt in (3, 9):     # Rook
        _PIECE_RAY_MASK[_pt] = _STRAIGHT_DIRS
    elif _pt in (4, 10):    # Queen
        _PIECE_RAY_MASK[_pt] = _STRAIGHT_DIRS | _DIAGONAL_DIRS
    else:
        _PIECE_RAY_MASK[_pt] = set()


@torch.no_grad()
def compute_spp_targets(
    boards: torch.Tensor,
    mask_builder: DynamicMaskBuilder,
) -> torch.Tensor:
    """
    Compute SPP (Square Property Prediction) targets from board tensors.

    Returns:
        (B, 64, 10) float tensor:
            channels [0:2] = [white_attack_count, black_attack_count] per square
            channels [2:10] = ray distances along 8 directions (N,NE,E,SE,S,SW,W,NW)
                              masked by piece type (non-sliders get all zeros)
    """
    B = boards.size(0)
    device = boards.device

    piece_types = extract_piece_types(boards)  # (B, 64) long, 0-12

    # --- Attack counts (channels 0-1) ---
    attack_mask = mask_builder.compute_attack_mask(boards)  # (B, 64, 64) bool
    # Zero out self-connections (diagonal)
    eye = torch.eye(64, dtype=torch.bool, device=device).unsqueeze(0)
    attack_mask = attack_mask & ~eye  # (B, 64, 64)

    # White pieces: types 0-5, Black pieces: types 6-11
    is_white = (piece_types < 6).unsqueeze(2)   # (B, 64, 1) â€” is piece on square i white?
    is_black = ((piece_types >= 6) & (piece_types < 12)).unsqueeze(2)

    # attack_mask[b, i, j] = piece on i attacks j
    # For target square j: white_count = sum over i where attack_mask[:, i, j] and piece is white
    white_attacks = (attack_mask & is_white).float().sum(dim=1)  # (B, 64) â€” sum over attackers dim
    black_attacks = (attack_mask & is_black).float().sum(dim=1)  # (B, 64)

    attack_counts = torch.stack([white_attacks, black_attacks], dim=-1)  # (B, 64, 2)

    # --- Ray distances (channels 2-9) ---
    # Occupancy: any piece present
    occupancy = boards[:, :12, :, :].sum(dim=1) > 0.5  # (B, 8, 8) bool

    ray_dists = torch.zeros(B, 64, 8, device=device)

    for d_idx, (dr, df) in enumerate(_RAY_DIRECTIONS):
        # For each step 1..7, check if (r + step*dr, f + step*df) is in-bounds and unoccupied
        # Accumulate: distance = number of steps before hitting blocker or edge
        # We track which squares are still "open" (haven't hit a blocker yet)
        still_open = torch.ones(B, 64, dtype=torch.bool, device=device)

        for step in range(1, 8):
            # Compute target coordinates for each source square
            # Source square sq: r = sq // 8, f = sq % 8
            src_r = torch.arange(64, device=device) // 8  # (64,)
            src_f = torch.arange(64, device=device) % 8

            tgt_r = src_r + step * dr  # (64,)
            tgt_f = src_f + step * df

            # In-bounds check
            in_bounds = (tgt_r >= 0) & (tgt_r < 8) & (tgt_f >= 0) & (tgt_f < 8)  # (64,)

            # For in-bounds squares, check if target is unoccupied
            # Clamp to valid indices for gather (we'll mask out-of-bounds later)
            safe_r = tgt_r.clamp(0, 7)
            safe_f = tgt_f.clamp(0, 7)

            # Gather occupancy at target positions: (B, 64)
            tgt_occupied = occupancy[:, safe_r, safe_f]  # (B, 64)

            # A square contributes distance if: in_bounds AND still_open AND not blocked
            # First, check if the target square is unoccupied (ray passes through)
            can_see = in_bounds.unsqueeze(0) & still_open & ~tgt_occupied  # (B, 64)

            # Increment distance for squares that are still open AND target is in-bounds
            # (even if blocked, the blocker square counts as "seen" distance)
            can_reach = in_bounds.unsqueeze(0) & still_open  # (B, 64)
            ray_dists[:, :, d_idx] += can_reach.float()

            # Close rays that hit a blocker or go out of bounds
            still_open = can_see  # only continue if unoccupied

    # Mask by piece type: zero out directions the piece can't use
    # Build piece-type ray mask: (13, 8) bool
    pt_ray_mask = torch.zeros(13, 8, dtype=torch.bool, device=device)
    for pt in range(13):
        for d_idx in _PIECE_RAY_MASK[pt]:
            pt_ray_mask[pt, d_idx] = True

    # Gather per-square mask: (B, 64, 8)
    sq_ray_mask = pt_ray_mask[piece_types]  # (B, 64, 8)
    ray_dists = ray_dists * sq_ray_mask.float()

    # Concatenate: (B, 64, 2) + (B, 64, 8) -> (B, 64, 10)
    return torch.cat([attack_counts, ray_dists], dim=-1)


# =============================================================================
# AdaLN â€” Adaptive Layer Normalization
# =============================================================================

class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on an external embedding.

    At initialization the projection is zero so AdaLN behaves as standard
    LayerNorm (scale=1, shift=0).
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(cond_dim, dim * 2)
        # Zero-init so conditioning starts as identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, S, dim)
            cond: (B, cond_dim)
        Returns:
            (B, S, dim) â€” normalized and modulated
        """
        scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


# =============================================================================
# Shared Layer-Conditioned Readout
# =============================================================================

class SharedLayerReadout(nn.Module):
    """
    Single shared readout producing layer-specific fusion tokens from perceiver
    latents and CSMP output, conditioned on layer fraction Ï„ âˆˆ [0, 1] via AdaLN.

    Called once per xattn layer with different Ï„ and text hidden states.
    Weights are shared across all calls; only the Fourier-encoded layer
    fraction differentiates behaviour.

    Flow per depth layer:
      0. (before depth loop) optionally modulate initial latents with causal recurrent text summary
      1. Cross-attend to text hidden states   (demand-driven query refinement)
      2. Cross-attend to policy latents       (high-level policy features, optional)
      3. Cross-attend to perceiver latents    (high-level chess features)
      4. Cross-attend to CSMP output          (low-level spatial, ungated)
      5. Self-attention among readout latents
      6. FFN
    """

    def __init__(
        self,
        num_latents: int = 16,
        perceiver_dim: int = 512,
        llm_dim: int = 2048,
        context_dim: int = 256,
        heads: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        depth: int = 1,
        fourier_dim: int = 64,
        max_freq: float = 10.0,
        use_text_conditioning: bool = False,
        use_policy_latent_cross_attention: bool = False,
        recurrent_text_state_enabled: bool = False,
        recurrent_text_state_dim: int = 256,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.perceiver_dim = perceiver_dim
        self.depth = depth
        self.use_text_conditioning = bool(use_text_conditioning)
        self.use_policy_latent_cross_attention = bool(use_policy_latent_cross_attention)
        self.recurrent_text_state_enabled = bool(recurrent_text_state_enabled)
        self.recurrent_text_state_dim = int(recurrent_text_state_dim)

        # --- Fourier encoding of layer fraction Ï„ ---
        self.register_buffer(
            'fourier_freqs',
            torch.linspace(1.0, max_freq, fourier_dim // 2),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(fourier_dim, perceiver_dim),
            nn.GELU(),
            nn.Linear(perceiver_dim, perceiver_dim),
        )

        # Learned latent queries (shared across all xattn layers)
        self.latents = nn.Parameter(torch.randn(1, num_latents, perceiver_dim) * 0.02)

        # Project LLM hidden states to perceiver_dim for text cross-attention KV
        self.text_kv_proj = nn.Linear(llm_dim, perceiver_dim, bias=False)

        # Project CSMP context to perceiver_dim if dimensions differ
        if context_dim != perceiver_dim:
            self.csmp_proj = nn.Linear(context_dim, perceiver_dim)
        else:
            self.csmp_proj = None

        # Optional recurrent text-state conditioning:
        # compute a causal text summary and use it to modulate initial fusion latents.
        if self.recurrent_text_state_enabled:
            if self.recurrent_text_state_dim <= 0:
                raise ValueError(
                    f"recurrent_text_state_dim must be > 0 (got {self.recurrent_text_state_dim})"
                )
            self.recurrent_text_norm = nn.LayerNorm(llm_dim)
            self.recurrent_text_gru = nn.GRU(
                input_size=llm_dim,
                hidden_size=self.recurrent_text_state_dim,
                num_layers=1,
                batch_first=True,
            )
            self.recurrent_to_latent_delta = nn.Linear(
                self.recurrent_text_state_dim,
                self.num_latents * self.perceiver_dim,
                bias=True,
            )
            # Start as exact no-op at forward init: x <- x + 0
            nn.init.zeros_(self.recurrent_to_latent_delta.weight)
            nn.init.zeros_(self.recurrent_to_latent_delta.bias)
            # Keep scale non-zero to avoid dead gradients when projection starts at zero.
            self.recurrent_latent_scale = nn.Parameter(torch.tensor(1.0))
        self._last_recurrent_state: Optional[torch.Tensor] = None
        self._last_recurrent_latent_delta_unscaled: Optional[torch.Tensor] = None
        self._last_recurrent_latent_delta: Optional[torch.Tensor] = None

        # --- Per-depth-layer modules ---
        # 1. Text cross-attention (queries attend to text KV)
        self.text_adaln_q = nn.ModuleList()
        self.text_norm_kv = nn.ModuleList()
        self.text_cross_attn = nn.ModuleList()

        # 2. Policy-latent cross-attention (optional, before perceiver branch)
        self.policy_adaln_q = nn.ModuleList()
        self.policy_norm_kv = nn.ModuleList()
        self.policy_cross_attn = nn.ModuleList()

        # 3. Perceiver cross-attention (queries attend to perceiver latents)
        self.perc_adaln_q = nn.ModuleList()
        self.perc_norm_kv = nn.ModuleList()
        self.perc_cross_attn = nn.ModuleList()

        # 4. CSMP cross-attention (queries attend to CSMP output, ungated)
        self.csmp_adaln_q = nn.ModuleList()
        self.csmp_norm_kv = nn.ModuleList()
        self.csmp_cross_attn = nn.ModuleList()

        # 5. Self-attention
        self.self_adaln = nn.ModuleList()
        self.self_attn = nn.ModuleList()

        # 6. FFN
        self.ffn_adaln = nn.ModuleList()
        self.ffn = nn.ModuleList()

        ffn_hidden = perceiver_dim * ffn_mult
        for _ in range(depth):
            # Text cross-attention
            self.text_adaln_q.append(AdaLN(perceiver_dim, perceiver_dim))
            self.text_norm_kv.append(nn.LayerNorm(perceiver_dim))
            self.text_cross_attn.append(ManualMultiHeadAttention(
                perceiver_dim, heads, dropout=dropout,
            ))

            # Policy-latent cross-attention (optional)
            if self.use_policy_latent_cross_attention:
                self.policy_adaln_q.append(AdaLN(perceiver_dim, perceiver_dim))
                self.policy_norm_kv.append(nn.LayerNorm(perceiver_dim))
                self.policy_cross_attn.append(ManualMultiHeadAttention(
                    perceiver_dim, heads, dropout=dropout,
                ))

            # Perceiver cross-attention
            self.perc_adaln_q.append(AdaLN(perceiver_dim, perceiver_dim))
            self.perc_norm_kv.append(nn.LayerNorm(perceiver_dim))
            self.perc_cross_attn.append(ManualMultiHeadAttention(
                perceiver_dim, heads, dropout=dropout,
            ))

            # CSMP cross-attention
            self.csmp_adaln_q.append(AdaLN(perceiver_dim, perceiver_dim))
            self.csmp_norm_kv.append(nn.LayerNorm(perceiver_dim))
            self.csmp_cross_attn.append(ManualMultiHeadAttention(
                perceiver_dim, heads, dropout=dropout,
            ))

            # Self-attention
            self.self_adaln.append(AdaLN(perceiver_dim, perceiver_dim))
            self.self_attn.append(ManualMultiHeadAttention(
                perceiver_dim, heads, dropout=dropout,
            ))

            # FFN
            self.ffn_adaln.append(AdaLN(perceiver_dim, perceiver_dim))
            self.ffn.append(nn.Sequential(
                nn.Linear(perceiver_dim, ffn_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden, perceiver_dim),
            ))

    def _fourier_encode(self, tau: torch.Tensor) -> torch.Tensor:
        """Encode layer fraction as Fourier features.

        Args:
            tau: (B,) float tensor in [0, 1]
        Returns:
            (B, fourier_dim) sinusoidal features
        """
        x = tau.unsqueeze(-1) * self.fourier_freqs * 2 * math.pi  # (B, F/2)
        return torch.cat([x.sin(), x.cos()], dim=-1)  # (B, F)

    def _summarize_text_causally(
        self,
        text_hidden_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Build one causal recurrent text summary per sample."""
        if not self.recurrent_text_state_enabled:
            return None

        B, S, _ = text_hidden_states.shape
        x = self.recurrent_text_norm(text_hidden_states)
        state_seq, _ = self.recurrent_text_gru(x)  # (B, S, D_state), causal over token axis

        if (
            text_attention_mask is not None
            and text_attention_mask.dim() == 2
            and text_attention_mask.shape[0] == B
            and text_attention_mask.shape[1] == S
        ):
            valid = text_attention_mask.to(device=state_seq.device, dtype=torch.bool)
            last_idx = valid.long().sum(dim=1) - 1
            last_idx = last_idx.clamp(min=0)
        else:
            last_idx = torch.full((B,), S - 1, device=state_seq.device, dtype=torch.long)

        state = state_seq[torch.arange(B, device=state_seq.device), last_idx, :]  # (B, D_state)
        self._last_recurrent_state = state.detach()
        return state

    def _build_recurrent_latent_delta(
        self,
        text_hidden_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Project recurrent text summary into an additive init delta for readout latents."""
        state = self._summarize_text_causally(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
        )
        if state is None:
            return None

        B = state.shape[0]
        delta_u = self.recurrent_to_latent_delta(state).view(
            B, self.num_latents, self.perceiver_dim
        )  # (B, N_lat, D)
        delta = torch.tanh(self.recurrent_latent_scale) * delta_u
        self._last_recurrent_latent_delta_unscaled = delta_u.detach()
        self._last_recurrent_latent_delta = delta.detach()
        return delta

    def forward(
        self,
        tau: float,
        text_hidden_states: torch.Tensor,
        perceiver_latents: torch.Tensor,
        csmp_output: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        policy_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tau:                layer_idx / total_llm_layers, float in [0, 1]
            text_hidden_states: (B, S, llm_dim) â€” current LLM layer hidden states
            perceiver_latents:  (B, N_lat, perceiver_dim)
            csmp_output:        (B, N_ctx, context_dim) â€” pre-Perceiver CSMP tokens
            policy_latents:     (B, N_policy, perceiver_dim) or None

        Returns:
            (B, num_latents, perceiver_dim) â€” layer-specific fusion tokens
        """
        B = perceiver_latents.size(0)

        # Compute layer conditioning
        tau_t = torch.full((B,), tau, device=perceiver_latents.device,
                           dtype=perceiver_latents.dtype)
        cond = self.cond_proj(self._fourier_encode(tau_t))  # (B, perceiver_dim)

        # Project inputs once
        x = self.latents.expand(B, -1, -1)  # (B, num_latents, perceiver_dim)
        if self.recurrent_text_state_enabled:
            latent_delta = self._build_recurrent_latent_delta(
                text_hidden_states=text_hidden_states,
                text_attention_mask=text_attention_mask,
            )
            if latent_delta is not None:
                x = x + latent_delta
        text_kv = self.text_kv_proj(text_hidden_states) if self.use_text_conditioning else None
        csmp_kv = self.csmp_proj(csmp_output) if self.csmp_proj is not None else csmp_output

        for d in range(self.depth):
            # 1. Optional text conditioning (disabled by default to avoid LM-token leakage)
            if self.use_text_conditioning:
                q = self.text_adaln_q[d](x, cond)
                kv = self.text_norm_kv[d](text_kv)

                text_attn_mask = None
                if (
                    text_attention_mask is not None
                    and text_attention_mask.dim() == 2
                    and text_attention_mask.shape[0] == B
                    and text_attention_mask.shape[1] == kv.shape[1]
                ):
                    # ManualMultiHeadAttention convention: True = masked-out
                    key_mask = ~text_attention_mask.to(device=kv.device, dtype=torch.bool)  # (B, K)
                    q_len = q.shape[1]
                    k_len = kv.shape[1]
                    n_heads = self.text_cross_attn[d].num_heads
                    text_attn_mask = key_mask.unsqueeze(1).expand(B, q_len, k_len)
                    text_attn_mask = (
                        text_attn_mask.unsqueeze(1)
                        .expand(B, n_heads, q_len, k_len)
                        .reshape(B * n_heads, q_len, k_len)
                    )

                ca_out, _ = self.text_cross_attn[d](
                    query=q,
                    key=kv,
                    value=kv,
                    attn_mask=text_attn_mask,
                    need_weights=False,
                )
                x = x + ca_out

            # 2. Optional cross-attend to policy latents (policy-specialized)
            if self.use_policy_latent_cross_attention and policy_latents is not None:
                q = self.policy_adaln_q[d](x, cond)
                kv = self.policy_norm_kv[d](policy_latents)
                ca_out, _ = self.policy_cross_attn[d](query=q, key=kv, value=kv, need_weights=False)
                x = x + ca_out

            # 3. Cross-attend to perceiver latents (high-level chess)
            q = self.perc_adaln_q[d](x, cond)
            kv = self.perc_norm_kv[d](perceiver_latents)
            ca_out, _ = self.perc_cross_attn[d](query=q, key=kv, value=kv, need_weights=False)
            x = x + ca_out

            # 4. Cross-attend to CSMP output (low-level spatial, no gate)
            q = self.csmp_adaln_q[d](x, cond)
            kv = self.csmp_norm_kv[d](csmp_kv)
            ca_out, _ = self.csmp_cross_attn[d](query=q, key=kv, value=kv, need_weights=False)
            x = x + ca_out

            # 5. Self-attention among readout latents
            x_n = self.self_adaln[d](x, cond)
            sa_out, _ = self.self_attn[d](query=x_n, key=x_n, value=x_n, need_weights=False)
            x = x + sa_out

            # 6. FFN
            x_n = self.ffn_adaln[d](x, cond)
            x = x + self.ffn[d](x_n)

        return x


# =============================================================================
# Gated Cross-Attention (v2 â€” shared readout + manual MHA + per-head gating)
# =============================================================================

class GatedCrossAttention(nn.Module):
    """
    Decoder-layer chess fusion module for injection into LLM decoder layers.

    Each instance holds:
      1. A reference to the shared ``SharedLayerReadout`` (set externally,
         not owned â€” avoids double-registering parameters).
      2. A recurrent text state used either for legacy recurrent-query
         cross-attention or for square-structured source mixing.
      3. Per-head tanh gating (initialized to 0 â†’ identity at init).
      4. FFN with reduced multiplier and scalar tanh gate.

    The ``tau`` attribute (layer fraction âˆˆ [0, 1]) is set once by the
    adapter during ``inject_into_llm``.
    """

    def __init__(
        self,
        llm_dim: int = 2048,
        perceiver_dim: int = 1280,
        context_dim: int = 256,
        num_fusion_tokens: int = 16,
        n_heads: int = 8,
        ffn_mult: int = 2,
        gate_init: float = 0.0,
        dropout: float = 0.1,
        recurrent_query_state_dim: int = 256,
        recurrent_query_use_mlp: bool = False,
        shared_recurrent_query_gru: Optional[nn.GRU] = None,
        xattn_mode: str = "recurrent_query_attn",
        structured_router_mode: str = "shared",
        text_gate_mode: str = "tanh_head",
    ):
        super().__init__()
        gate_init = float(gate_init)
        self.llm_dim = llm_dim
        self.context_dim = int(context_dim)
        self.n_heads = n_heads
        self.head_dim = llm_dim // n_heads
        self.num_fusion_tokens = int(num_fusion_tokens)
        self.recurrent_query_state_dim = int(recurrent_query_state_dim)
        self.recurrent_query_use_mlp = bool(recurrent_query_use_mlp)
        self.xattn_mode = str(xattn_mode)
        self.structured_router_mode = str(structured_router_mode)
        self.text_gate_mode = str(text_gate_mode)
        self._shared_recurrent_query_gru_ref: Optional[weakref.ReferenceType] = None
        assert llm_dim % n_heads == 0, f"llm_dim {llm_dim} not divisible by n_heads {n_heads}"
        if self.xattn_mode not in {"recurrent_query_attn", "structured_square_mixer"}:
            raise ValueError(
                "xattn_mode must be one of "
                "{'recurrent_query_attn', 'structured_square_mixer'} "
                f"(got {self.xattn_mode!r})"
            )
        if self.structured_router_mode not in {"shared", "per_head"}:
            raise ValueError(
                "structured_router_mode must be one of "
                "{'shared', 'per_head'} "
                f"(got {self.structured_router_mode!r})"
            )
        if self.text_gate_mode not in {"none", "tanh_head"}:
            raise ValueError(
                "text_gate_mode must be one of "
                "{'none', 'tanh_head'} "
                f"(got {self.text_gate_mode!r})"
            )

        # --- Shared readout reference (set by ChessFusionAdapter) ---
        self.tau: float = 0.0
        self._shared_readout: Optional[SharedLayerReadout] = None

        # Shared output projection from either recurrent cross-attn or
        # structured square mixing back into the LLM residual stream.
        self.q_norm = None
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = nn.Linear(llm_dim, llm_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Per-head gating: (1, n_heads, 1, 1) â€” broadcasts over batch, seq, head_dim
        self.gate = nn.Parameter(torch.full((1, n_heads, 1, 1), gate_init))

        # Canonical x-attn mode:
        # GRU(text) -> queries, then direct cross-attention to policy/perceiver/csmp.
        if self.recurrent_query_state_dim <= 0:
            raise ValueError(
                f"recurrent_query_state_dim must be > 0 (got {self.recurrent_query_state_dim})"
            )
        if abs(float(gate_init)) < 1e-12:
            print(
                "  [XAttn recurrent-query] gate_init is 0.0; recurrent-query path gradients "
                "to Q/K/V/GRU begin only after head gates open."
            )
        self.recurrent_query_norm = nn.LayerNorm(llm_dim)
        if shared_recurrent_query_gru is not None:
            if shared_recurrent_query_gru.input_size != llm_dim:
                raise ValueError(
                    f"shared_recurrent_query_gru.input_size ({shared_recurrent_query_gru.input_size}) "
                    f"!= llm_dim ({llm_dim})"
                )
            if shared_recurrent_query_gru.hidden_size != self.recurrent_query_state_dim:
                raise ValueError(
                    f"shared_recurrent_query_gru.hidden_size ({shared_recurrent_query_gru.hidden_size}) "
                    f"!= recurrent_query_state_dim ({self.recurrent_query_state_dim})"
                )
            self._shared_recurrent_query_gru_ref = weakref.ref(shared_recurrent_query_gru)
            self.recurrent_query_gru = None
        else:
            self.recurrent_query_gru = nn.GRU(
                input_size=llm_dim,
                hidden_size=self.recurrent_query_state_dim,
                num_layers=1,
                batch_first=True,
            )
        if self.xattn_mode == "recurrent_query_attn":
            self.recurrent_query_proj = self._build_recurrent_head(llm_dim)
            self.perc_kv_norm = nn.LayerNorm(perceiver_dim)
            self.perc_k_proj = nn.Linear(perceiver_dim, llm_dim, bias=False)
            self.perc_v_proj = nn.Linear(perceiver_dim, llm_dim, bias=False)

            self.csmp_kv_norm = nn.LayerNorm(self.context_dim)
            self.csmp_k_proj = nn.Linear(self.context_dim, llm_dim, bias=False)
            self.csmp_v_proj = nn.Linear(self.context_dim, llm_dim, bias=False)

            self.policy_kv_norm = nn.LayerNorm(perceiver_dim)
            self.policy_k_proj = nn.Linear(perceiver_dim, llm_dim, bias=False)
            self.policy_v_proj = nn.Linear(perceiver_dim, llm_dim, bias=False)

            # Learned source blend gates. Init > 0 so all sources are available
            # once xattn head gates begin to open.
            self.source_gate_perc = nn.Parameter(torch.tensor(1.0))
            self.source_gate_csmp = nn.Parameter(torch.tensor(1.0))
            self.source_gate_policy = nn.Parameter(torch.tensor(1.0))

            self.structured_csmp_square_mlp = None
            self.structured_perceiver_square_mlp = None
            self.structured_policy_square_mlp = None
            self.structured_global_perceiver_mlp = None
            self.structured_global_side_mlp = None
            self.structured_router_stem = None
            self.structured_square_weight_proj = None
            self.structured_global_weight_proj = None
            self.text_gate_mlp = None
        else:
            self.recurrent_query_proj = None
            self.perc_kv_norm = None
            self.perc_k_proj = None
            self.perc_v_proj = None
            self.csmp_kv_norm = None
            self.csmp_k_proj = None
            self.csmp_v_proj = None
            self.policy_kv_norm = None
            self.policy_k_proj = None
            self.policy_v_proj = None
            self.register_parameter("source_gate_perc", None)
            self.register_parameter("source_gate_csmp", None)
            self.register_parameter("source_gate_policy", None)

            self.structured_csmp_square_mlp = self._build_source_value_mlp(self.context_dim, dropout)
            self.structured_perceiver_square_mlp = self._build_source_value_mlp(perceiver_dim, dropout)
            self.structured_policy_square_mlp = self._build_source_value_mlp(perceiver_dim, dropout)
            self.structured_global_perceiver_mlp = self._build_source_value_mlp(perceiver_dim, dropout)
            self.structured_global_side_mlp = self._build_source_value_mlp(self.context_dim, dropout)
            self.structured_router_stem = self._build_router_stem(
                self.recurrent_query_state_dim + perceiver_dim,
                dropout,
            )
            square_router_dim = 64 * 3
            global_router_dim = 2
            if self.structured_router_mode == "per_head":
                square_router_dim *= self.n_heads
                global_router_dim *= self.n_heads
            self.structured_square_weight_proj = nn.Linear(self.llm_dim, square_router_dim)
            self.structured_global_weight_proj = nn.Linear(self.llm_dim, global_router_dim)
            if self.text_gate_mode == "tanh_head":
                self.text_gate_mlp = nn.Linear(self.llm_dim, self.n_heads)
                nn.init.zeros_(self.text_gate_mlp.weight)
                nn.init.zeros_(self.text_gate_mlp.bias)
            else:
                self.text_gate_mlp = None

        self._last_recurrent_query_state: Optional[torch.Tensor] = None
        self._last_source_weights: Optional[torch.Tensor] = None
        self._last_structured_metrics: Optional[Dict[str, torch.Tensor]] = None
        self._last_structured_square_sparse_loss: Optional[torch.Tensor] = None
        self._last_structured_square_usage_entropy_norm: Optional[torch.Tensor] = None
        self._last_structured_gate_usage_mean_abs: Optional[torch.Tensor] = None
        self._capture_last_token_trace: bool = False
        self._last_token_trace: Optional[Dict[str, torch.Tensor]] = None

        # --- FFN ---
        self.ffn_norm = nn.LayerNorm(llm_dim)
        ffn_hidden = llm_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, llm_dim),
        )
        self.ffn_gate = nn.Parameter(torch.tensor(gate_init))

        self._last_text_gate: Optional[torch.Tensor] = None
        self._last_recurrent_state: Optional[torch.Tensor] = None
        self._last_recurrent_latent_delta_unscaled: Optional[torch.Tensor] = None
        self._last_recurrent_latent_delta: Optional[torch.Tensor] = None

    def _build_recurrent_head(self, output_dim: int) -> nn.Module:
        return self._build_conditioning_head(self.recurrent_query_state_dim, output_dim)

    def _build_conditioning_head(self, input_dim: int, output_dim: int) -> nn.Module:
        if self.recurrent_query_use_mlp:
            return nn.Sequential(
                nn.Linear(input_dim, self.llm_dim),
                nn.GELU(),
                nn.Linear(self.llm_dim, output_dim),
            )
        return nn.Linear(input_dim, output_dim)

    def _build_source_value_mlp(self, input_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.llm_dim, self.llm_dim),
        )

    def _build_router_stem(self, input_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _require_rank3(name: str, tensor: torch.Tensor) -> None:
        if tensor.dim() != 3:
            raise ValueError(f"Expected {name} shape (B, N, D); got {tuple(tensor.shape)}")

    @staticmethod
    def _require_square_count(name: str, tensor: torch.Tensor, expected: int = 64) -> None:
        if tensor.size(1) != expected:
            raise ValueError(f"Expected {name} to have {expected} square tokens, got {tensor.size(1)}")

    @staticmethod
    def _build_valid_token_mask(
        text_attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            text_attention_mask is not None
            and text_attention_mask.dim() == 2
            and text_attention_mask.shape[0] == batch_size
            and text_attention_mask.shape[1] == seq_len
        ):
            return text_attention_mask.to(device=device, dtype=torch.bool)
        return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    def _get_recurrent_query_gru(self) -> nn.GRU:
        if self._shared_recurrent_query_gru_ref is not None:
            shared = self._shared_recurrent_query_gru_ref()
            if shared is None:
                raise RuntimeError("Shared recurrent-query GRU reference is no longer valid.")
            return shared
        if self.recurrent_query_gru is None:
            raise RuntimeError("recurrent_query_gru is not initialized.")
        return self.recurrent_query_gru

    def _reshape_values_to_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, num_slots, _ = values.shape
        return values.view(batch_size, num_slots, self.n_heads, self.head_dim)

    def _compute_effective_structured_gate(
        self,
        valid: torch.Tensor,
        token_gate_logits: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        static_gate_logits = self.gate.view(1, 1, self.n_heads)
        if token_gate_logits is None:
            effective_gate = torch.tanh(static_gate_logits).expand(valid.size(0), valid.size(1), -1)
        else:
            effective_gate = torch.tanh(static_gate_logits + token_gate_logits)
        return effective_gate.to(dtype=dtype) * valid.unsqueeze(-1).to(dtype=dtype)

    def _source_cross_attn(
        self,
        q: torch.Tensor,
        kv_source: torch.Tensor,
        norm: nn.LayerNorm,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
    ) -> torch.Tensor:
        """Cross-attend recurrent queries to one source."""
        B, _, _, _ = q.shape
        k = k_proj(norm(kv_source))
        v = v_proj(norm(kv_source))
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

    def set_last_token_trace_capture(self, enabled: bool) -> None:
        self._capture_last_token_trace = bool(enabled)
        if not self._capture_last_token_trace:
            self._last_token_trace = None

    def clear_last_token_trace(self) -> None:
        self._last_token_trace = None

    def _cache_structured_metrics(
        self,
        slot_weights: torch.Tensor,
        global_weights: torch.Tensor,
        token_gate_logits: Optional[torch.Tensor],
        effective_gates: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
    ) -> None:
        valid = self._build_valid_token_mask(
            text_attention_mask,
            batch_size=slot_weights.size(0),
            seq_len=slot_weights.size(1),
            device=slot_weights.device,
        )
        valid_f = valid.to(dtype=slot_weights.dtype)
        denom = valid_f.sum().clamp(min=1.0)

        if slot_weights.dim() == 3:
            slot_weights_per_head = slot_weights.unsqueeze(2)
        elif slot_weights.dim() == 4:
            slot_weights_per_head = slot_weights
        else:
            raise ValueError(
                "Expected structured slot_weights rank 3 or 4, got "
                f"{tuple(slot_weights.shape)}"
            )

        if global_weights.dim() == 3:
            global_weights_per_head = global_weights.unsqueeze(2)
        elif global_weights.dim() == 4:
            global_weights_per_head = global_weights
        else:
            raise ValueError(
                "Expected structured global_weights rank 3 or 4, got "
                f"{tuple(global_weights.shape)}"
            )

        router_heads = int(slot_weights_per_head.size(2))
        denom_per_head = denom * float(router_heads)

        slot_mean_per_head = (
            slot_weights_per_head * valid_f.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=(0, 1)) / denom
        slot_mean = slot_mean_per_head.mean(dim=0)
        slot_probs = slot_weights_per_head.clamp_min(1e-12)
        slot_entropy = (
            -(slot_probs * slot_probs.log()).sum(dim=-1) * valid_f.unsqueeze(-1)
        ).sum() / denom_per_head
        source_mass_per_head = slot_mean_per_head.view(router_heads, 3, 64).sum(dim=-1)
        source_mass = source_mass_per_head.mean(dim=0)

        square_weights_per_head = slot_weights_per_head.view(
            slot_weights_per_head.size(0),
            slot_weights_per_head.size(1),
            router_heads,
            3,
            64,
        ).sum(dim=3)
        square_mean_per_head = (
            square_weights_per_head * valid_f.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=(0, 1)) / denom
        square_mean = square_mean_per_head.mean(dim=0)
        square_probs = square_weights_per_head.clamp_min(1e-12)
        square_entropy = (
            -(square_probs * square_probs.log()).sum(dim=-1) * valid_f.unsqueeze(-1)
        ).sum() / denom_per_head
        square_entropy_norm = square_entropy / max(math.log(float(square_weights_per_head.size(-1))), 1e-12)
        square_usage_entropy_per_head = -(
            square_mean_per_head.clamp_min(1e-12) * square_mean_per_head.clamp_min(1e-12).log()
        ).sum(dim=-1)
        square_usage_entropy = square_usage_entropy_per_head.mean()
        square_usage_entropy_norm = square_usage_entropy / max(
            math.log(float(square_mean_per_head.size(-1))),
            1e-12,
        )

        max_slot_index = torch.argmax(slot_mean)
        max_slot_source_index = torch.div(max_slot_index, 64, rounding_mode="floor")
        max_slot_square_index = torch.remainder(max_slot_index, 64)
        max_square_index = torch.argmax(square_mean)

        global_mean_per_head = (
            global_weights_per_head * valid_f.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=(0, 1)) / denom
        global_mean = global_mean_per_head.mean(dim=0)
        global_probs = global_weights_per_head.clamp_min(1e-12)
        global_entropy = (
            -(global_probs * global_probs.log()).sum(dim=-1) * valid_f.unsqueeze(-1)
        ).sum() / denom_per_head

        valid_gate_mask = valid.unsqueeze(-1).expand_as(effective_gates)
        masked_effective_gates = effective_gates.masked_select(valid_gate_mask)
        effective_gate_abs_mean_per_head = (
            effective_gates.abs() * valid_f.unsqueeze(-1)
        ).sum(dim=(0, 1)) / denom
        if masked_effective_gates.numel() > 0:
            effective_gate_abs_mean = masked_effective_gates.abs().mean()
            effective_gate_std = masked_effective_gates.std(unbiased=False)
        else:
            zero = effective_gates.new_tensor(0.0)
            effective_gate_abs_mean = zero
            effective_gate_std = zero

        if token_gate_logits is not None:
            masked_token_gate_logits = token_gate_logits.masked_select(valid_gate_mask)
            token_gate_logit_mean_per_head = (
                token_gate_logits * valid_f.unsqueeze(-1)
            ).sum(dim=(0, 1)) / denom
            token_gate_logit_mean = (
                masked_token_gate_logits.mean()
                if masked_token_gate_logits.numel() > 0
                else token_gate_logits.new_tensor(0.0)
            )
        else:
            token_gate_logit_mean_per_head = effective_gates.new_zeros(self.n_heads)
            token_gate_logit_mean = effective_gates.new_tensor(0.0)

        self._last_structured_square_sparse_loss = square_entropy_norm
        self._last_structured_square_usage_entropy_norm = square_usage_entropy_norm
        self._last_structured_gate_usage_mean_abs = effective_gate_abs_mean
        self._last_structured_metrics = {
            "router_mode": self.structured_router_mode,
            "slot_mean": slot_mean.detach(),
            "slot_mean_per_head": slot_mean_per_head.detach(),
            "slot_entropy": slot_entropy.detach(),
            "source_mass": source_mass.detach(),
            "source_mass_per_head": source_mass_per_head.detach(),
            "max_slot_index": max_slot_index.detach(),
            "max_slot_source_index": max_slot_source_index.detach(),
            "max_slot_square_index": max_slot_square_index.detach(),
            "max_slot_mass": slot_mean[max_slot_index].detach(),
            "square_mean": square_mean.detach(),
            "square_mean_per_head": square_mean_per_head.detach(),
            "square_entropy": square_entropy.detach(),
            "square_entropy_norm": square_entropy_norm.detach(),
            "square_usage_entropy": square_usage_entropy.detach(),
            "square_usage_entropy_norm": square_usage_entropy_norm.detach(),
            "max_square_index": max_square_index.detach(),
            "max_square_mass": square_mean[max_square_index].detach(),
            "global_mean": global_mean.detach(),
            "global_mean_per_head": global_mean_per_head.detach(),
            "global_entropy": global_entropy.detach(),
            "effective_gate_abs_mean": effective_gate_abs_mean.detach(),
            "effective_gate_abs_mean_per_head": effective_gate_abs_mean_per_head.detach(),
            "effective_gate_std": effective_gate_std.detach(),
            "token_gate_logit_mean": token_gate_logit_mean.detach(),
            "token_gate_logit_mean_per_head": token_gate_logit_mean_per_head.detach(),
        }

    def _cache_last_token_trace(
        self,
        slot_weights: torch.Tensor,
        global_weights: torch.Tensor,
        token_gate_logits: Optional[torch.Tensor],
        effective_gates: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor],
        square_values_heads: torch.Tensor,
        global_values_heads: torch.Tensor,
    ) -> None:
        if not self._capture_last_token_trace:
            self._last_token_trace = None
            return

        batch_size = int(slot_weights.size(0))
        seq_len = int(slot_weights.size(1))
        if seq_len <= 0:
            self._last_token_trace = None
            return

        valid = self._build_valid_token_mask(
            text_attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=slot_weights.device,
        )
        valid_counts = valid.long().sum(dim=1)
        last_token_indices = (valid_counts - 1).clamp(min=0, max=seq_len - 1)
        batch_indices = torch.arange(batch_size, device=slot_weights.device)
        selected_effective_gates = effective_gates[batch_indices, last_token_indices].detach()
        selected_token_gate_logits = None
        if token_gate_logits is not None:
            selected_token_gate_logits = token_gate_logits[batch_indices, last_token_indices].detach()

        trace: Dict[str, Any] = {
            "router_mode": self.structured_router_mode,
            "effective_head_gates": selected_effective_gates,
            "last_token_indices": last_token_indices.detach(),
        }
        if selected_token_gate_logits is not None:
            trace["token_gate_logits"] = selected_token_gate_logits

        o_proj_weight = self.o_proj.weight.detach().float().view(self.llm_dim, self.n_heads, self.head_dim)
        square_values_projected = torch.einsum(
            "bnhd,ohd->bnho",
            square_values_heads.detach().float(),
            o_proj_weight,
        ).permute(0, 2, 1, 3)
        global_values_projected = torch.einsum(
            "bghd,ohd->bgho",
            global_values_heads.detach().float(),
            o_proj_weight,
        ).permute(0, 2, 1, 3)

        if slot_weights.dim() == 3:
            raw_slot_weights = slot_weights[batch_indices, last_token_indices].detach()
            global_token_weights = global_weights[batch_indices, last_token_indices].detach()
            source_square_weights = raw_slot_weights.view(batch_size, 3, 64)
            aggregate_square_weights = source_square_weights.sum(dim=1)
            head_square_weights = raw_slot_weights.unsqueeze(1).expand(-1, self.n_heads, -1)
            head_global_weights = global_token_weights.unsqueeze(1).expand(-1, self.n_heads, -1)
            trace.update(
                {
                    "raw_slot_weights": raw_slot_weights,
                    "source_square_weights": source_square_weights,
                    "aggregate_square_weights": aggregate_square_weights,
                    "global_weights": global_token_weights,
                }
            )
        else:
            raw_slot_weights_per_head = slot_weights[batch_indices, last_token_indices].detach()
            global_weights_per_head = global_weights[batch_indices, last_token_indices].detach()
            source_square_weights_per_head = raw_slot_weights_per_head.view(batch_size, self.n_heads, 3, 64)
            aggregate_square_weights_per_head = source_square_weights_per_head.sum(dim=2)
            head_square_weights = raw_slot_weights_per_head
            head_global_weights = global_weights_per_head
            trace.update(
                {
                    "raw_slot_weights": raw_slot_weights_per_head.mean(dim=1),
                    "source_square_weights": source_square_weights_per_head.mean(dim=1),
                    "aggregate_square_weights": aggregate_square_weights_per_head.mean(dim=1),
                    "global_weights": global_weights_per_head.mean(dim=1),
                    "raw_slot_weights_per_head": raw_slot_weights_per_head,
                    "source_square_weights_per_head": source_square_weights_per_head,
                    "aggregate_square_weights_per_head": aggregate_square_weights_per_head,
                    "global_weights_per_head": global_weights_per_head,
                }
            )

        selected_effective_gates_f = selected_effective_gates.detach().float()
        head_square_weights_f = head_square_weights.detach().float()
        head_global_weights_f = head_global_weights.detach().float()
        gate_scale = selected_effective_gates_f.unsqueeze(-1).unsqueeze(-1)

        square_contrib_vectors_per_head = (
            gate_scale
            * head_square_weights_f.unsqueeze(-1)
            * square_values_projected[batch_indices]
        ).view(batch_size, self.n_heads, 3, 64, self.llm_dim)
        global_contrib_vectors_per_head = (
            gate_scale
            * head_global_weights_f.unsqueeze(-1)
            * global_values_projected[batch_indices]
        )

        source_square_contribution_norms_per_head = square_contrib_vectors_per_head.norm(dim=-1)
        aggregate_square_contribution_norms_per_head = source_square_contribution_norms_per_head.sum(dim=2)
        global_contribution_norms_per_head = global_contrib_vectors_per_head.norm(dim=-1)

        source_square_contribution_vectors = square_contrib_vectors_per_head.sum(dim=1)
        global_contribution_vectors = global_contrib_vectors_per_head.sum(dim=1)
        source_square_contribution_norms = source_square_contribution_vectors.norm(dim=-1)
        aggregate_square_contribution_norms = source_square_contribution_norms.sum(dim=1)
        global_contribution_norms = global_contribution_vectors.norm(dim=-1)

        contribution_total = (
            source_square_contribution_norms.sum(dim=(1, 2))
            + global_contribution_norms.sum(dim=-1)
        ).clamp_min(1e-12)
        source_square_contribution_norms = (
            source_square_contribution_norms
            / contribution_total.view(batch_size, 1, 1)
        )
        aggregate_square_contribution_norms = source_square_contribution_norms.sum(dim=1)
        global_contribution_norms = (
            global_contribution_norms
            / contribution_total.view(batch_size, 1)
        )

        per_head_contribution_total = (
            source_square_contribution_norms_per_head.sum(dim=(2, 3))
            + global_contribution_norms_per_head.sum(dim=-1)
        ).clamp_min(1e-12)
        source_square_contribution_norms_per_head = (
            source_square_contribution_norms_per_head
            / per_head_contribution_total.unsqueeze(-1).unsqueeze(-1)
        )
        aggregate_square_contribution_norms_per_head = (
            source_square_contribution_norms_per_head.sum(dim=2)
        )
        global_contribution_norms_per_head = (
            global_contribution_norms_per_head
            / per_head_contribution_total.unsqueeze(-1)
        )

        trace.update(
            {
                "source_square_contribution_norms": source_square_contribution_norms.detach(),
                "aggregate_square_contribution_norms": aggregate_square_contribution_norms.detach(),
                "global_contribution_norms": global_contribution_norms.detach(),
            }
        )
        if self.structured_router_mode == "per_head":
            trace.update(
                {
                    "source_square_contribution_norms_per_head": (
                        source_square_contribution_norms_per_head.detach()
                    ),
                    "aggregate_square_contribution_norms_per_head": (
                        aggregate_square_contribution_norms_per_head.detach()
                    ),
                    "global_contribution_norms_per_head": (
                        global_contribution_norms_per_head.detach()
                    ),
                }
            )

        self._last_token_trace = trace

    def _forward_recurrent_query_mode(
        self,
        hidden_states: torch.Tensor,
        perceiver_latents: torch.Tensor,
        context: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
        policy_latents: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Direct recurrent-query xattn over policy/perceiver/csmp sources."""
        self._last_token_trace = None
        B, S, _ = hidden_states.shape

        rq_in = self.recurrent_query_norm(hidden_states)
        rq_state, _ = self._get_recurrent_query_gru()(rq_in)  # (B, S, D_state)
        if (
            text_attention_mask is not None
            and text_attention_mask.dim() == 2
            and text_attention_mask.shape[0] == B
            and text_attention_mask.shape[1] == S
        ):
            valid = text_attention_mask.to(device=rq_state.device, dtype=torch.bool)
            rq_state = rq_state.masked_fill(~valid.unsqueeze(-1), 0.0)
        self._last_recurrent_query_state = rq_state.detach()

        q = self.recurrent_query_proj(rq_state).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        out_terms: List[torch.Tensor] = []
        weight_terms: List[torch.Tensor] = []

        # Perceiver source
        w_perc = torch.tanh(self.source_gate_perc)
        out_perc = self._source_cross_attn(
            q=q,
            kv_source=perceiver_latents,
            norm=self.perc_kv_norm,
            k_proj=self.perc_k_proj,
            v_proj=self.perc_v_proj,
        )
        out_terms.append(w_perc * out_perc)
        weight_terms.append(w_perc.detach())

        # CSMP source (if available)
        if context is not None and context.dim() == 3:
            w_csmp = torch.tanh(self.source_gate_csmp)
            out_csmp = self._source_cross_attn(
                q=q,
                kv_source=context,
                norm=self.csmp_kv_norm,
                k_proj=self.csmp_k_proj,
                v_proj=self.csmp_v_proj,
            )
            out_terms.append(w_csmp * out_csmp)
            weight_terms.append(w_csmp.detach())

        # Policy source (optional)
        if policy_latents is not None and policy_latents.dim() == 3:
            w_policy = torch.tanh(self.source_gate_policy)
            out_policy = self._source_cross_attn(
                q=q,
                kv_source=policy_latents,
                norm=self.policy_kv_norm,
                k_proj=self.policy_k_proj,
                v_proj=self.policy_v_proj,
            )
            out_terms.append(w_policy * out_policy)
            weight_terms.append(w_policy.detach())

        if not out_terms:
            return hidden_states

        self._last_source_weights = torch.stack(weight_terms)
        attn_out = torch.stack(out_terms, dim=0).sum(dim=0)  # (B, H, S, D_h)
        attn_out = torch.tanh(self.gate) * attn_out
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.llm_dim)
        attn_out = self.o_proj(attn_out)
        hidden_states = hidden_states + attn_out

        residual = hidden_states
        ffn_out = self.ffn(self.ffn_norm(hidden_states))
        hidden_states = residual + torch.tanh(self.ffn_gate) * ffn_out
        return hidden_states

    def _forward_structured_square_mixer_mode(
        self,
        hidden_states: torch.Tensor,
        perceiver_latents: torch.Tensor,
        context: Optional[torch.Tensor],
        csmp_square_tokens: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
        policy_latents: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.structured_router_stem is None
            or self.structured_square_weight_proj is None
            or self.structured_global_weight_proj is None
        ):
            raise RuntimeError("structured_square_mixer mode is not initialized on this layer.")

        self._last_structured_metrics = None
        self._last_structured_square_sparse_loss = None
        self._last_structured_square_usage_entropy_norm = None
        self._last_structured_gate_usage_mean_abs = None
        self._last_token_trace = None
        self._require_rank3("perceiver_latents", perceiver_latents)
        if perceiver_latents.size(1) < 65:
            raise ValueError(
                "structured_square_mixer requires perceiver_latents with 65 tokens "
                f"(64 squares + global), got {perceiver_latents.size(1)}"
            )
        if context is None:
            raise ValueError("structured_square_mixer requires context so the side token global path is available")
        self._require_rank3("context", context)
        if context.size(1) < 1:
            raise ValueError("structured_square_mixer requires context to include the side token")
        if csmp_square_tokens is None:
            raise ValueError("structured_square_mixer requires csmp_square_tokens")
        if policy_latents is None:
            raise ValueError("structured_square_mixer requires policy_latents")

        self._require_rank3("csmp_square_tokens", csmp_square_tokens)
        self._require_square_count("csmp_square_tokens", csmp_square_tokens, expected=64)
        self._require_rank3("policy_latents", policy_latents)
        self._require_square_count("policy_latents", policy_latents, expected=64)

        B, S, _ = hidden_states.shape
        rq_in = self.recurrent_query_norm(hidden_states)
        rq_state, _ = self._get_recurrent_query_gru()(rq_in)  # (B, S, D_state)
        valid = self._build_valid_token_mask(text_attention_mask, batch_size=B, seq_len=S, device=rq_state.device)
        rq_state = rq_state.masked_fill(~valid.unsqueeze(-1), 0.0)
        self._last_recurrent_query_state = rq_state.detach()
        self._last_source_weights = None

        global_perceiver_latent = perceiver_latents[:, 64, :]
        global_perceiver_cond = global_perceiver_latent.unsqueeze(1).expand(-1, S, -1)
        router_inputs = torch.cat([rq_state, global_perceiver_cond], dim=-1)
        router_hidden = self.structured_router_stem(router_inputs)
        square_logits = self.structured_square_weight_proj(router_hidden)
        global_logits = self.structured_global_weight_proj(router_hidden)
        token_gate_logits: Optional[torch.Tensor]
        if self.text_gate_mlp is not None:
            token_gate_logits = self.text_gate_mlp(router_hidden)
        else:
            token_gate_logits = None

        square_values = torch.cat(
            [
                self.structured_csmp_square_mlp(csmp_square_tokens),
                self.structured_perceiver_square_mlp(perceiver_latents[:, :64, :]),
                self.structured_policy_square_mlp(policy_latents),
            ],
            dim=1,
        )  # (B, 192, llm_dim)
        square_values_heads = self._reshape_values_to_heads(square_values)

        global_values = torch.stack(
            [
                self.structured_global_perceiver_mlp(global_perceiver_latent),
                self.structured_global_side_mlp(context[:, -1, :]),
            ],
            dim=1,
        )  # (B, 2, llm_dim)
        global_values_heads = self._reshape_values_to_heads(global_values)

        if self.structured_router_mode == "per_head":
            square_logits = square_logits.view(B, S, self.n_heads, 64 * 3)
            square_weights = torch.softmax(square_logits.float(), dim=-1).to(dtype=hidden_states.dtype)
            square_mix = torch.einsum("bshn,bnhd->bshd", square_weights, square_values_heads)

            global_logits = global_logits.view(B, S, self.n_heads, 2)
            global_weights = torch.softmax(global_logits.float(), dim=-1).to(dtype=hidden_states.dtype)
            global_mix = torch.einsum("bshg,bghd->bshd", global_weights, global_values_heads)
        else:
            square_weights = torch.softmax(square_logits.float(), dim=-1).to(dtype=hidden_states.dtype)
            square_mix = torch.einsum("bsn,bnhd->bshd", square_weights, square_values_heads)

            global_weights = torch.softmax(global_logits.float(), dim=-1).to(dtype=hidden_states.dtype)
            global_mix = torch.einsum("bsg,bghd->bshd", global_weights, global_values_heads)

        effective_gates = self._compute_effective_structured_gate(
            valid=valid,
            token_gate_logits=token_gate_logits,
            dtype=hidden_states.dtype,
        )
        self._last_text_gate = (
            token_gate_logits.detach()
            if token_gate_logits is not None
            else None
        )

        self._cache_structured_metrics(
            slot_weights=square_weights.float(),
            global_weights=global_weights.float(),
            token_gate_logits=token_gate_logits.float() if token_gate_logits is not None else None,
            effective_gates=effective_gates.float(),
            text_attention_mask=text_attention_mask,
        )
        self._cache_last_token_trace(
            slot_weights=square_weights.float(),
            global_weights=global_weights.float(),
            token_gate_logits=token_gate_logits.float() if token_gate_logits is not None else None,
            effective_gates=effective_gates.float(),
            text_attention_mask=text_attention_mask,
            square_values_heads=square_values_heads,
            global_values_heads=global_values_heads,
        )

        attn_out = (square_mix + global_mix) * effective_gates.unsqueeze(-1)
        attn_out = attn_out.contiguous().view(B, S, self.llm_dim)
        attn_out = self.o_proj(attn_out)
        hidden_states = hidden_states + attn_out

        residual = hidden_states
        ffn_out = self.ffn(self.ffn_norm(hidden_states))
        hidden_states = residual + torch.tanh(self.ffn_gate) * ffn_out
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        perceiver_latents: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        csmp_square_tokens: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        policy_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:     (B, S, llm_dim) from LLM layer
            perceiver_latents: (B, N_lat, perceiver_dim) shared Perceiver output
            context:           (B, S_ctx, context_dim) pre-Perceiver CSMP output
            csmp_square_tokens:(B, 64, context_dim) square-aligned CSMP tokens
            policy_latents:    (B, N_policy, perceiver_dim) policy readout latents

        Returns:
            (B, S, llm_dim) gated cross-attended hidden states
        """
        if self.xattn_mode == "structured_square_mixer":
            return self._forward_structured_square_mixer_mode(
                hidden_states=hidden_states,
                perceiver_latents=perceiver_latents,
                context=context,
                csmp_square_tokens=csmp_square_tokens,
                text_attention_mask=text_attention_mask,
                policy_latents=policy_latents,
            )

        self._last_structured_metrics = None
        self._last_structured_square_sparse_loss = None
        self._last_structured_square_usage_entropy_norm = None
        self._last_structured_gate_usage_mean_abs = None
        self._last_text_gate = None
        return self._forward_recurrent_query_mode(
            hidden_states=hidden_states,
            perceiver_latents=perceiver_latents,
            context=context,
            text_attention_mask=text_attention_mask,
            policy_latents=policy_latents,
        )


class PrependLatentReadout(nn.Module):
    """
    Readout that projects learned latent queries into LLM text space.

    Modes:
      - cross_attn:
          learned query tokens cross-attend to CSMP, Perceiver, and policy sources
      - structured_mlp:
          one shared shallow MLP consumes the aligned [CSMP_i, Perceiver_i, Policy_i]
          square latents and emits one text latent per square
    """

    def __init__(
        self,
        num_prepend_latents: int,
        perceiver_dim: int,
        context_dim: int,
        llm_dim: int,
        heads: int,
        policy_dim: Optional[int] = None,
        mode: str = "cross_attn",
        structured_mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_prepend_latents = int(num_prepend_latents)
        self.perceiver_dim = int(perceiver_dim)
        self.context_dim = int(context_dim)
        self.llm_dim = int(llm_dim)
        self.policy_dim = int(policy_dim) if policy_dim is not None else self.perceiver_dim
        self.mode = str(mode)

        if self.mode not in {"cross_attn", "structured_mlp"}:
            raise ValueError(
                "PrependLatentReadout mode must be one of "
                "{'cross_attn', 'structured_mlp'} "
                f"(got {self.mode!r})"
            )

        if self.mode == "cross_attn":
            self.query_latents = nn.Parameter(
                torch.randn(1, self.num_prepend_latents, self.perceiver_dim) * 0.02
            )
            self.query_norm = nn.LayerNorm(self.perceiver_dim)
            self.csmp_norm = nn.LayerNorm(self.context_dim)
            self.csmp_to_perceiver = nn.Linear(self.context_dim, self.perceiver_dim, bias=False)
            self.perceiver_norm = nn.LayerNorm(self.perceiver_dim)
            self.policy_norm = nn.LayerNorm(self.policy_dim)
            self.policy_to_perceiver = (
                nn.Linear(self.policy_dim, self.perceiver_dim, bias=False)
                if self.policy_dim != self.perceiver_dim
                else nn.Identity()
            )
            self.cross_attn = ManualMultiHeadAttention(
                embed_dim=self.perceiver_dim,
                num_heads=heads,
                dropout=dropout,
            )
            self.ffn_norm = nn.LayerNorm(self.perceiver_dim)
            self.ffn = nn.Sequential(
                nn.Linear(self.perceiver_dim, self.perceiver_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.perceiver_dim * 2, self.perceiver_dim),
            )
            self.to_llm = nn.Linear(self.perceiver_dim, self.llm_dim, bias=False)
        else:
            if self.num_prepend_latents != 64:
                raise ValueError(
                    "structured_mlp prepend mode requires num_prepend_latents=64 "
                    f"(got {self.num_prepend_latents})"
                )
            hidden_dim = int(structured_mlp_hidden_dim or self.llm_dim)
            concat_dim = self.context_dim + self.perceiver_dim + self.policy_dim
            self.csmp_square_norm = nn.LayerNorm(self.context_dim)
            self.perceiver_square_norm = nn.LayerNorm(self.perceiver_dim)
            self.policy_square_norm = nn.LayerNorm(self.policy_dim)
            self.square_mlp = nn.Sequential(
                nn.LayerNorm(concat_dim),
                nn.Linear(concat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.llm_dim),
            )

    @staticmethod
    def _require_rank3(name: str, tensor: torch.Tensor) -> None:
        if tensor.dim() != 3:
            raise ValueError(f"Expected {name} shape (B, N, D); got {tuple(tensor.shape)}")

    @staticmethod
    def _require_square_count(name: str, tensor: torch.Tensor, expected: int = 64) -> None:
        if tensor.size(1) != expected:
            raise ValueError(f"Expected {name} to have {expected} square tokens, got {tensor.size(1)}")

    def _forward_cross_attn(
        self,
        perceiver_latents: torch.Tensor,
        csmp_square_tokens: Optional[torch.Tensor],
        policy_latents: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self._require_rank3("perceiver_latents", perceiver_latents)
        if perceiver_latents.size(1) != 65:
            raise ValueError(
                "Prepend latent cross-attn mode expects 65 Perceiver latents "
                "(64 squares + 1 global), "
                f"got {perceiver_latents.size(1)}"
            )

        kv_parts = []
        if csmp_square_tokens is not None:
            self._require_rank3("csmp_square_tokens", csmp_square_tokens)
            self._require_square_count("csmp_square_tokens", csmp_square_tokens, expected=64)
            kv_parts.append(self.csmp_to_perceiver(self.csmp_norm(csmp_square_tokens)))

        kv_parts.append(self.perceiver_norm(perceiver_latents))

        if policy_latents is not None:
            self._require_rank3("policy_latents", policy_latents)
            self._require_square_count("policy_latents", policy_latents, expected=64)
            kv_parts.append(self.policy_to_perceiver(self.policy_norm(policy_latents)))

        batch_size = perceiver_latents.size(0)
        x = self.query_latents.expand(batch_size, -1, -1)
        q = self.query_norm(x)
        kv = torch.cat(kv_parts, dim=1)
        ca_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )
        x = x + ca_out
        x = x + self.ffn(self.ffn_norm(x))
        return self.to_llm(x)

    def _forward_structured_mlp(
        self,
        perceiver_latents: torch.Tensor,
        csmp_square_tokens: Optional[torch.Tensor],
        policy_latents: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self._require_rank3("perceiver_latents", perceiver_latents)
        if perceiver_latents.size(1) < 64:
            raise ValueError(
                "structured_mlp prepend mode expects at least 64 Perceiver square latents, "
                f"got {perceiver_latents.size(1)}"
            )
        if csmp_square_tokens is None:
            raise ValueError("structured_mlp prepend mode requires csmp_square_tokens")
        if policy_latents is None:
            raise ValueError("structured_mlp prepend mode requires policy_latents")

        self._require_rank3("csmp_square_tokens", csmp_square_tokens)
        self._require_square_count("csmp_square_tokens", csmp_square_tokens, expected=64)
        self._require_rank3("policy_latents", policy_latents)
        self._require_square_count("policy_latents", policy_latents, expected=64)

        concat = torch.cat(
            [
                self.csmp_square_norm(csmp_square_tokens),
                self.perceiver_square_norm(perceiver_latents[:, :64, :]),
                self.policy_square_norm(policy_latents),
            ],
            dim=-1,
        )
        return self.square_mlp(concat)

    def forward(
        self,
        perceiver_latents: torch.Tensor,
        csmp_square_tokens: Optional[torch.Tensor] = None,
        policy_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "structured_mlp":
            return self._forward_structured_mlp(
                perceiver_latents=perceiver_latents,
                csmp_square_tokens=csmp_square_tokens,
                policy_latents=policy_latents,
            )
        return self._forward_cross_attn(
            perceiver_latents=perceiver_latents,
            csmp_square_tokens=csmp_square_tokens,
            policy_latents=policy_latents,
        )


class LayerPseudotokenAttention(nn.Module):
    """
    Independent learned KV memory for one LLM layer.

    This is intentionally decoupled from Fusion readout/x-attn gates so
    pseudotokens can learn even when gated cross-attention is frozen or gated off.
    """

    def __init__(
        self,
        llm_dim: int,
        n_heads: int,
        num_tokens: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.n_heads = n_heads
        self.head_dim = llm_dim // n_heads
        self.num_tokens = int(num_tokens)
        assert llm_dim % n_heads == 0, f"llm_dim {llm_dim} not divisible by n_heads {n_heads}"

        self.attn_dropout = nn.Dropout(dropout)

        if self.num_tokens > 0:
            self.pseudo_k = nn.Parameter(torch.randn(1, self.num_tokens, llm_dim) * 0.02)
            self.pseudo_v = nn.Parameter(torch.randn(1, self.num_tokens, llm_dim) * 0.02)
        else:
            self.register_parameter('pseudo_k', None)
            self.register_parameter('pseudo_v', None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pseudo_k is None or self.pseudo_v is None:
            return hidden_states

        B, S, _ = hidden_states.shape
        q = hidden_states
        k = self.pseudo_k.expand(B, -1, -1)
        v = self.pseudo_v.expand(B, -1, -1)

        if k.dtype != q.dtype:
            k = k.to(dtype=q.dtype)
        if v.dtype != q.dtype:
            v = v.to(dtype=q.dtype)

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.llm_dim)
        return hidden_states + attn_out


# =============================================================================
# Fusion Decoder Layer Wrapper
# =============================================================================

class FusionDecoderLayer(nn.Module):
    """
    Wraps an original LLM decoder layer and optionally appends:
    - per-layer pseudotoken attention
    - gated cross-attention to chess context
    """

    def __init__(
        self,
        original_layer: nn.Module,
        gated_xattn: Optional[GatedCrossAttention] = None,
        pseudotoken_attn: Optional[LayerPseudotokenAttention] = None,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.gated_xattn = gated_xattn
        self.pseudotoken_attn = pseudotoken_attn
        # Stored by the adapter before each forward pass
        self._perceiver_latents: Optional[torch.Tensor] = None
        self._context: Optional[torch.Tensor] = None
        self._csmp_square_tokens: Optional[torch.Tensor] = None
        self._text_attention_mask: Optional[torch.Tensor] = None
        self._policy_latents: Optional[torch.Tensor] = None

    def set_chess_context(self, perceiver_latents: torch.Tensor,
                          context: Optional[torch.Tensor] = None,
                          csmp_square_tokens: Optional[torch.Tensor] = None,
                          text_attention_mask: Optional[torch.Tensor] = None,
                          policy_latents: Optional[torch.Tensor] = None):
        self._perceiver_latents = perceiver_latents
        self._context = context
        self._csmp_square_tokens = csmp_square_tokens
        self._text_attention_mask = text_attention_mask
        self._policy_latents = policy_latents

    def clear_chess_context(self):
        self._perceiver_latents = None
        self._context = None
        self._csmp_square_tokens = None
        self._text_attention_mask = None
        self._policy_latents = None

    # Class-level profiling accumulators (reset per LLM forward by training loop).
    # Stored as lists (one entry per fusion layer call) to enable per-layer breakdown.
    # Train loop reads these via sum() for backward-compat with scalar usage.
    _profile_enabled = False
    _profile_original_ms: list = []
    _profile_xattn_ms: list = []
    _profile_count = 0

    @classmethod
    def reset_profile(cls):
        cls._profile_original_ms = []
        cls._profile_xattn_ms = []
        cls._profile_count = 0

    def forward(self, *args, **kwargs):
        _p = FusionDecoderLayer._profile_enabled

        if _p:
            import time
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        # Run original layer
        outputs = self.original_layer(*args, **kwargs)

        # Apply independent per-layer pseudotoken KV memory
        if self.pseudotoken_attn is not None:
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                hidden_states = self.pseudotoken_attn(hidden_states)
                outputs = (hidden_states,) + outputs[1:]
            else:
                outputs = self.pseudotoken_attn(outputs)

        if _p:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()

        # Apply gated cross-attention if perceiver latents are set
        if self.gated_xattn is not None and self._perceiver_latents is not None:
            # outputs is typically a tuple (hidden_states, ...) for HF models
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                hidden_states = self.gated_xattn(
                    hidden_states,
                    self._perceiver_latents,
                    self._context,
                    self._csmp_square_tokens,
                    self._text_attention_mask,
                    self._policy_latents,
                )
                outputs = (hidden_states,) + outputs[1:]
            else:
                outputs = self.gated_xattn(
                    outputs,
                    self._perceiver_latents,
                    self._context,
                    self._csmp_square_tokens,
                    self._text_attention_mask,
                    self._policy_latents,
                )

        if _p:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()
            FusionDecoderLayer._profile_original_ms.append((_t1 - _t0) * 1000)
            FusionDecoderLayer._profile_xattn_ms.append((_t2 - _t1) * 1000)
            FusionDecoderLayer._profile_count += 1

        return outputs


# =============================================================================
# Maia Fusion Adapter (Top-Level)
# =============================================================================

class ChessFusionAdapter(nn.Module):
    """
    Top-level adapter for the Chess-Fusion architecture.

    Combines:
    - Maia2 backbone (CNN + Transformer)
    - Multi-scale feature extractor (CNN + mid-transformer + final)
    - Square Latent Encoder (32 latents -> dual branches)
    - Gated cross-attention modules (for LLM injection)

    The adapter produces:
    - Perceiver latents for per-layer gated cross-attention readout at specified LLM layers
    - Auxiliary losses (policy + eval)
    """

    def __init__(self, config, llm_dim: int = 2048, llm_num_heads: int = 16):
        super().__init__()

        # Extract fusion config
        if hasattr(config, 'chess_fusion'):
            cfg = config.chess_fusion
        else:
            from training.config import ChessFusionConfig
            cfg = ChessFusionConfig()

        self.cfg = cfg
        self.llm_dim = llm_dim
        self.llm_num_heads = llm_num_heads
        self.use_cnn = getattr(cfg, 'use_cnn', True)
        self.backbone_init = str(getattr(cfg, 'backbone_init', 'maia_pretrained'))
        self.structured_latents = bool(getattr(cfg, 'structured_latents', False))
        self.square_heads_include_global_latent = bool(
            getattr(cfg, 'square_heads_include_global_latent', True)
        )
        self.enable_lm_prepend_latents = bool(getattr(cfg, 'enable_lm_prepend_latents', False))
        self.num_lm_prepend_latents = (
            int(getattr(cfg, 'num_lm_prepend_latents', 16))
            if self.enable_lm_prepend_latents
            else 0
        )
        self.lm_prepend_latent_mode = str(getattr(cfg, 'lm_prepend_latent_mode', 'cross_attn'))
        self.lm_prepend_structured_mlp_hidden_dim = getattr(
            cfg, 'lm_prepend_structured_mlp_hidden_dim', None
        )
        self.lm_prepend_latents_use_positional_encoding = bool(
            getattr(cfg, 'lm_prepend_latents_use_positional_encoding', True)
        )
        self.enable_lm_pseudotokens = bool(getattr(cfg, 'enable_lm_pseudotokens', True))
        self.num_lm_pseudotokens = int(getattr(cfg, 'num_lm_pseudotokens', 0)) if self.enable_lm_pseudotokens else 0
        self.xattn_mode = str(getattr(cfg, 'xattn_mode', 'recurrent_query_attn'))
        self.xattn_structured_router_mode = str(getattr(cfg, 'xattn_structured_router_mode', 'shared'))
        self.xattn_text_gate_mode = str(getattr(cfg, 'xattn_text_gate_mode', 'tanh_head'))
        self.xattn_recurrent_query_share_gru_across_layers = bool(
            getattr(cfg, 'xattn_recurrent_query_share_gru_across_layers', False)
        )

        self.shared_recurrent_query_gru: Optional[nn.GRU] = None
        if self.xattn_recurrent_query_share_gru_across_layers:
            rq_dim = int(getattr(cfg, 'xattn_recurrent_query_state_dim', 256))
            self.shared_recurrent_query_gru = nn.GRU(
                input_size=self.llm_dim,
                hidden_size=rq_dim,
                num_layers=1,
                batch_first=True,
            )

        if self.backbone_init not in {'maia_pretrained', 'random'}:
            raise ValueError(
                "chess_fusion.backbone_init must be one of {'maia_pretrained', 'random'} "
                f"(got {self.backbone_init!r})"
            )

        if self.structured_latents and cfg.num_latents != 65:
            raise ValueError(
                f"chess_fusion.structured_latents=True requires num_latents=65 (got {cfg.num_latents})"
            )
        if self.enable_lm_prepend_latents and self.num_lm_prepend_latents <= 0:
            raise ValueError(
                "chess_fusion.enable_lm_prepend_latents=True requires "
                f"num_lm_prepend_latents > 0 (got {self.num_lm_prepend_latents})"
            )
        if self.enable_lm_prepend_latents and cfg.num_latents != 65:
            raise ValueError(
                "chess_fusion.enable_lm_prepend_latents=True expects perceiver num_latents=65 "
                f"(got {cfg.num_latents})"
            )
        if self.enable_lm_prepend_latents and self.lm_prepend_latent_mode not in {'cross_attn', 'structured_mlp'}:
            raise ValueError(
                "chess_fusion.lm_prepend_latent_mode must be one of "
                "{'cross_attn', 'structured_mlp'} "
                f"(got {self.lm_prepend_latent_mode!r})"
            )
        if self.enable_lm_prepend_latents and self.lm_prepend_latent_mode == 'structured_mlp':
            if self.num_lm_prepend_latents != 64:
                raise ValueError(
                    "chess_fusion.lm_prepend_latent_mode='structured_mlp' requires "
                    f"num_lm_prepend_latents=64 (got {self.num_lm_prepend_latents})"
                )
            if not bool(getattr(cfg, 'use_chess_structure_mp', False)):
                raise ValueError(
                    "chess_fusion.lm_prepend_latent_mode='structured_mlp' requires "
                    "use_chess_structure_mp=True"
                )
            if not bool(getattr(cfg, 'use_structured_policy_head', False)):
                raise ValueError(
                    "chess_fusion.lm_prepend_latent_mode='structured_mlp' requires "
                    "use_structured_policy_head=True"
                )
        if self.xattn_mode not in {'recurrent_query_attn', 'structured_square_mixer'}:
            raise ValueError(
                "chess_fusion.xattn_mode must be one of "
                "{'recurrent_query_attn', 'structured_square_mixer'} "
                f"(got {self.xattn_mode!r})"
            )
        if self.xattn_structured_router_mode not in {'shared', 'per_head'}:
            raise ValueError(
                "chess_fusion.xattn_structured_router_mode must be one of "
                "{'shared', 'per_head'} "
                f"(got {self.xattn_structured_router_mode!r})"
            )
        if self.xattn_text_gate_mode not in {'none', 'tanh_head'}:
            raise ValueError(
                "chess_fusion.xattn_text_gate_mode must be one of "
                "{'none', 'tanh_head'} "
                f"(got {self.xattn_text_gate_mode!r})"
            )
        if self.xattn_mode == 'structured_square_mixer':
            if not self.structured_latents:
                raise ValueError(
                    "chess_fusion.xattn_mode='structured_square_mixer' requires structured_latents=True"
                )
            if cfg.num_latents != 65:
                raise ValueError(
                    "chess_fusion.xattn_mode='structured_square_mixer' requires num_latents=65 "
                    f"(got {cfg.num_latents})"
                )
            if not bool(getattr(cfg, 'use_chess_structure_mp', False)):
                raise ValueError(
                    "chess_fusion.xattn_mode='structured_square_mixer' requires use_chess_structure_mp=True"
                )
            if not bool(getattr(cfg, 'use_structured_policy_head', False)):
                raise ValueError(
                    "chess_fusion.xattn_mode='structured_square_mixer' requires use_structured_policy_head=True"
                )
        self._warn_ignored_structured_readout_settings()
        # Store freeze flags as instance attributes for live control
        self.freeze_cnn = cfg.freeze_cnn
        self.freeze_transformer = cfg.freeze_transformer
        self.freeze_csmp = getattr(cfg, 'freeze_csmp', False)
        self.freeze_perceiver = getattr(cfg, 'freeze_perceiver', False)
        self.freeze_xattn = cfg.freeze_xattn
        self.freeze_prepend_latents = getattr(cfg, 'freeze_prepend_latents', False)
        self.freeze_lm_pseudotokens = getattr(cfg, 'freeze_lm_pseudotokens', False)

        # Maia backbone (only when CNN is enabled)
        if self.use_cnn:
            class _BackboneConfig:
                class maia:
                    model_type = cfg.model_type
                    elo_self = cfg.elo_self
                    elo_oppo = cfg.elo_oppo
                    freeze_backbone = False

            self.backbone = MaiaPolicyModel(_BackboneConfig())
            if self.backbone_init == 'random':
                self._reinitialize_student_backbone()

            # Freeze Maia prediction heads (not used in fusion path)
            for head_name in ("fc_1", "fc_2", "fc_3", "fc_3_1"):
                head = getattr(self.backbone.maia, head_name, None)
                if head is not None:
                    for p in head.parameters():
                        p.requires_grad = False

            # Objective Maia teacher (always pretrained, always frozen).
            # Distillation targets come from this model, not the trainable student backbone.
            self.teacher_backbone = MaiaPolicyModel(_BackboneConfig())
            self.teacher_backbone.eval()
            for p in self.teacher_backbone.parameters():
                p.requires_grad = False
            print("  Policy teacher: objective frozen Maia backbone (independent from student backbone)")
        else:
            self.backbone = None
            self.teacher_backbone = None
            print("  [ChessFusion] CNN disabled â€” board-only mode (pos + piece embeddings)")

        # Build CSMP config dict if enabled
        csmp_config = None
        if getattr(cfg, 'use_chess_structure_mp', False):
            csmp_config = {
                'csmp_dim': getattr(cfg, 'csmp_dim', 1024),
                'csmp_pos_dim': getattr(cfg, 'csmp_pos_dim', 32),
                'csmp_piece_dim': getattr(cfg, 'csmp_piece_dim', 64),
                'csmp_cnn_proj_dim': getattr(cfg, 'csmp_cnn_proj_dim', 0),
                'csmp_layers': getattr(cfg, 'csmp_layers', 4),
                'csmp_heads': getattr(cfg, 'csmp_heads', 8),
                'csmp_ffn_mult': getattr(cfg, 'csmp_ffn_mult', 2),
                'csmp_dropout': getattr(cfg, 'csmp_dropout', 0.1),
                'csmp_use_ray_mask': getattr(cfg, 'csmp_use_ray_mask', True),
                'csmp_use_attack_mask': getattr(cfg, 'csmp_use_attack_mask', True),
                'csmp_use_xy_coords': getattr(cfg, 'csmp_use_xy_coords', False),
                'csmp_relative_mode': getattr(cfg, 'csmp_relative_mode', 'none'),
                'csmp_relative_edge_dim': getattr(cfg, 'csmp_relative_edge_dim', 16),
                'csmp_ablation_no_mask': getattr(cfg, 'csmp_ablation_no_mask', False),
            }

        # Multi-scale feature extractor
        self.multi_scale = MultiScaleFeatureExtractor(
            self.backbone,
            tap_dim=cfg.tap_projection_dim,
            cnn_tap_layers=getattr(cfg, 'cnn_tap_layers', None),
            concat_cnn_taps=getattr(cfg, 'concat_cnn_taps', False),
            use_transformer_taps=cfg.use_transformer_taps,
            use_cnn=self.use_cnn,
            csmp_config=csmp_config,
        )

        # Square Latent Encoder
        self.perceiver = SquareLatentEncoder(
            tap_dim=cfg.tap_projection_dim,
            perceiver_dim=cfg.perceiver_dim,
            num_latents=cfg.num_latents,
            depth=cfg.perceiver_depth,
            heads=cfg.perceiver_heads,
            num_fusion_tokens=cfg.num_fusion_tokens,
            num_eval_buckets=cfg.num_eval_buckets,
            enable_eval_head=bool(getattr(cfg, "aux_eval_weight", 0.0) > 0),
            use_engineered_concat=cfg.use_engineered_concat,
            dropout=getattr(cfg, 'perceiver_dropout', 0.1),
            structured_latents=self.structured_latents,
            latent_context_mask_type=getattr(cfg, 'latent_context_mask_type', 'full'),
            global_latent_attends_all=getattr(cfg, 'global_latent_attends_all', True),
            square_latent_attends_side_token=getattr(cfg, 'square_latent_attends_side_token', True),
            use_structured_policy_head=getattr(cfg, 'use_structured_policy_head', False),
            policy_include_global_latent=self.square_heads_include_global_latent,
            structured_policy_query_layers=getattr(cfg, 'structured_policy_query_layers', 4),
            structured_policy_query_heads=getattr(cfg, 'structured_policy_query_heads', None),
            structured_policy_ffn_mult=getattr(cfg, 'structured_policy_ffn_mult', 2),
            structured_policy_use_move_bias=getattr(cfg, 'structured_policy_use_move_bias', True),
        )

        # Shared layer-conditioned readout (legacy recurrent-query mode only)
        self.xattn_layer_indices = list(cfg.xattn_layers)
        raw_pseudotoken_layers = getattr(cfg, 'lm_pseudotoken_layers', None)
        if raw_pseudotoken_layers is None:
            self.pseudotoken_layer_indices = list(self.xattn_layer_indices)
        else:
            self.pseudotoken_layer_indices = [int(i) for i in raw_pseudotoken_layers]
        if not self.enable_lm_pseudotokens or self.num_lm_pseudotokens <= 0:
            self.pseudotoken_layer_indices = []
        self.shared_readout: Optional[SharedLayerReadout] = None
        if self.xattn_mode == 'recurrent_query_attn':
            self.shared_readout = SharedLayerReadout(
                num_latents=cfg.num_fusion_tokens,
                perceiver_dim=cfg.perceiver_dim,
                llm_dim=self.llm_dim,
                context_dim=cfg.tap_projection_dim,
                heads=cfg.perceiver_heads,
                ffn_mult=getattr(cfg, 'xattn_ffn_mult', 2),
                dropout=getattr(cfg, 'xattn_dropout', 0.1),
                depth=getattr(cfg, 'readout_depth', 1),
                fourier_dim=getattr(cfg, 'shared_readout_fourier_dim', 64),
                use_text_conditioning=False,
                use_policy_latent_cross_attention=bool(
                    getattr(cfg, 'readout_use_policy_latent_cross_attention', False)
                ),
                recurrent_text_state_enabled=False,
                recurrent_text_state_dim=256,
            )

        self.prepend_latent_readout: Optional[PrependLatentReadout] = None
        if self.enable_lm_prepend_latents:
            self.prepend_latent_readout = PrependLatentReadout(
                num_prepend_latents=self.num_lm_prepend_latents,
                perceiver_dim=cfg.perceiver_dim,
                context_dim=cfg.tap_projection_dim,
                llm_dim=self.llm_dim,
                heads=cfg.perceiver_heads,
                policy_dim=cfg.perceiver_dim,
                mode=self.lm_prepend_latent_mode,
                structured_mlp_hidden_dim=self.lm_prepend_structured_mlp_hidden_dim,
                dropout=getattr(cfg, 'xattn_dropout', 0.1),
            )

        # Gated cross-attention modules (created here, injected into LLM later)
        # Each module references the shared readout (set in inject_into_llm).
        self.gated_xattns = nn.ModuleList([
            GatedCrossAttention(
                llm_dim=self.llm_dim,
                perceiver_dim=cfg.perceiver_dim,
                context_dim=cfg.tap_projection_dim,
                num_fusion_tokens=cfg.num_fusion_tokens,
                n_heads=cfg.xattn_heads,
                ffn_mult=getattr(cfg, 'xattn_ffn_mult', 2),
                gate_init=cfg.xattn_gate_init,
                dropout=getattr(cfg, 'xattn_dropout', 0.1),
                recurrent_query_state_dim=int(getattr(cfg, 'xattn_recurrent_query_state_dim', 256)),
                recurrent_query_use_mlp=bool(getattr(cfg, 'xattn_recurrent_query_use_mlp', False)),
                shared_recurrent_query_gru=self.shared_recurrent_query_gru,
                xattn_mode=self.xattn_mode,
                structured_router_mode=self.xattn_structured_router_mode,
                text_gate_mode=self.xattn_text_gate_mode,
            )
            for _ in self.xattn_layer_indices
        ])

        self.lm_pseudotoken_layers = nn.ModuleList([
            LayerPseudotokenAttention(
                llm_dim=self.llm_dim,
                n_heads=self.llm_num_heads,
                num_tokens=self.num_lm_pseudotokens,
                dropout=getattr(cfg, 'xattn_dropout', 0.1),
            )
            for _ in self.pseudotoken_layer_indices
        ])

        # --- BSR / SPP self-supervised auxiliary heads ---
        # Always create heads so they can be enabled/disabled mid-training
        # via live controller weight changes. Only run in forward() when weight > 0.
        self.bsr_head = AuxSquareHead(
            perceiver_dim=cfg.perceiver_dim,
            head_dim=getattr(cfg, 'bsr_dim', 256),
            output_dim=13,  # 12 piece types + empty
            n_heads=getattr(cfg, 'bsr_heads', 4),
            n_layers=getattr(cfg, 'bsr_layers', 2),
            dropout=getattr(cfg, 'bsr_dropout', 0.1),
            structured_mode=self.structured_latents,
            include_global_latent=self.square_heads_include_global_latent,
        )
        bsr_w = getattr(cfg, 'bsr_weight', 0.0)
        print(f"  BSR head: dim={cfg.bsr_dim}, heads={cfg.bsr_heads}, "
              f"layers={cfg.bsr_layers}, weight={bsr_w}{' (disabled)' if bsr_w == 0 else ''}")

        self.spp_head = AuxSquareHead(
            perceiver_dim=cfg.perceiver_dim,
            head_dim=getattr(cfg, 'spp_dim', 256),
            output_dim=10,  # 2 attack counts + 8 ray distances
            n_heads=getattr(cfg, 'spp_heads', 4),
            n_layers=getattr(cfg, 'spp_layers', 2),
            dropout=getattr(cfg, 'spp_dropout', 0.1),
            structured_mode=self.structured_latents,
            include_global_latent=self.square_heads_include_global_latent,
        )
        self.spp_mask_builder = DynamicMaskBuilder()
        spp_w = getattr(cfg, 'spp_weight', 0.0)
        print(f"  SPP head: dim={cfg.spp_dim}, heads={cfg.spp_heads}, "
              f"layers={cfg.spp_layers}, weight={spp_w}{' (disabled)' if spp_w == 0 else ''}")

        # Apply backbone freezing (only when backbone exists)
        if self.use_cnn and (self.freeze_cnn or self.freeze_transformer):
            self._apply_freezing()

        if self.freeze_csmp:
            self._freeze_csmp_params()

        if self.freeze_perceiver:
            self._freeze_perceiver_params()

        if self.freeze_xattn:
            self._freeze_xattn_params()

        if self.freeze_prepend_latents:
            self._freeze_prepend_latent_params()

        if self.freeze_lm_pseudotokens:
            self._freeze_lm_pseudotoken_params()

        # Track wrapped decoder layers for pseudotoken and/or x-attn injection
        self._fusion_layers: List[FusionDecoderLayer] = []

        self._print_summary()

    def _print_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nChessFusionAdapter Summary:")
        print(f"  Total params:     {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Student backbone init: {self.backbone_init}")
        print(f"  Objective policy teacher: {self.teacher_backbone is not None}")
        print(f"  Perceiver: {self.cfg.num_latents} latents x {self.cfg.perceiver_depth} layers")
        print(f"  XAttn mode: {self.xattn_mode}")
        if self.xattn_mode == "structured_square_mixer":
            print(
                f"  Structured router: mode={self.xattn_structured_router_mode}, "
                f"text_gate_mode={self.xattn_text_gate_mode}"
            )
        if self.shared_readout is not None:
            print(f"  Fusion tokens: {self.cfg.num_fusion_tokens} per xattn layer (policy_latent_cond={bool(getattr(self.cfg, 'readout_use_policy_latent_cross_attention', False))})")
        else:
            print("  Fusion tokens: unused in structured_square_mixer mode")
        print(
            f"  XAttn recurrent-query: state_dim={int(getattr(self.cfg, 'xattn_recurrent_query_state_dim', 256))}, "
            f"use_mlp={bool(getattr(self.cfg, 'xattn_recurrent_query_use_mlp', False))}, "
            f"share_gru_across_layers={self.xattn_recurrent_query_share_gru_across_layers}"
        )
        if self.enable_lm_prepend_latents:
            print(
                f"  LM prepend latents: {self.num_lm_prepend_latents} tokens "
                f"(llm_pos_enc={self.lm_prepend_latents_use_positional_encoding})"
            )
        else:
            print("  LM prepend latents: disabled")
        if self.enable_lm_pseudotokens:
            print(f"  LM pseudotokens: {self.num_lm_pseudotokens} per pseudotoken layer")
            print(f"  LM pseudotoken layers: {self.pseudotoken_layer_indices}")
        else:
            print("  LM pseudotokens: disabled")
        print(f"  Gated X-Attn at LLM layers: {self.xattn_layer_indices}")
        print(f"  Freeze CNN: {self.freeze_cnn}, Transformer: {self.freeze_transformer}, CSMP: {self.freeze_csmp}, Perceiver: {self.freeze_perceiver}, XAttn: {self.freeze_xattn}, Pseudotokens: {self.freeze_lm_pseudotokens}")
        print(f"  Aux weights: policy={self.cfg.aux_policy_weight}, eval={self.cfg.aux_eval_weight}")

    def _warn_ignored_structured_readout_settings(self) -> None:
        if self.xattn_mode != 'structured_square_mixer':
            return

        ignored: List[str] = []
        if int(getattr(self.cfg, 'num_fusion_tokens', 16)) != 16:
            ignored.append(f"num_fusion_tokens={int(getattr(self.cfg, 'num_fusion_tokens', 16))}")
        if int(getattr(self.cfg, 'readout_depth', 1)) != 1:
            ignored.append(f"readout_depth={int(getattr(self.cfg, 'readout_depth', 1))}")
        if int(getattr(self.cfg, 'shared_readout_fourier_dim', 64)) != 64:
            ignored.append(
                f"shared_readout_fourier_dim={int(getattr(self.cfg, 'shared_readout_fourier_dim', 64))}"
            )
        if bool(getattr(self.cfg, 'readout_use_policy_latent_cross_attention', False)):
            ignored.append("readout_use_policy_latent_cross_attention=True")

        if ignored:
            print(
                "  WARNING: chess_fusion.xattn_mode='structured_square_mixer' ignores "
                f"shared readout settings: {', '.join(ignored)}"
            )

    def _reinitialize_student_backbone(self):
        """Randomly reinitialize student backbone parameters."""
        if self.backbone is None:
            return
        reset_count = 0
        for module in self.backbone.maia.modules():
            reset = getattr(module, 'reset_parameters', None)
            if callable(reset):
                reset()
                reset_count += 1
        print(f"  Student backbone randomly initialized (reset modules: {reset_count})")

    # â”€â”€ Freezing Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _apply_freezing(self):
        """Apply initial freezing based on config."""
        if self.freeze_cnn:
            self._freeze_cnn_params()
        if self.freeze_transformer:
            self._freeze_transformer_params()

    def _freeze_cnn_params(self):
        maia = self.backbone.maia
        for p in maia.chess_cnn.parameters():
            p.requires_grad = False
        for p in maia.to_patch_embedding.parameters():
            p.requires_grad = False
        maia.pos_embedding.requires_grad = False

    def _unfreeze_cnn_params(self):
        maia = self.backbone.maia
        for p in maia.chess_cnn.parameters():
            p.requires_grad = True
        for p in maia.to_patch_embedding.parameters():
            p.requires_grad = True
        maia.pos_embedding.requires_grad = True

    def _freeze_transformer_params(self):
        maia = self.backbone.maia
        for p in maia.transformer.parameters():
            p.requires_grad = False
        for p in maia.elo_embedding.parameters():
            p.requires_grad = False
        for p in maia.last_ln.parameters():
            p.requires_grad = False

    def _unfreeze_transformer_params(self):
        maia = self.backbone.maia
        for p in maia.transformer.parameters():
            p.requires_grad = True
        for p in maia.elo_embedding.parameters():
            p.requires_grad = True
        for p in maia.last_ln.parameters():
            p.requires_grad = True

    def _freeze_csmp_params(self):
        if hasattr(self.multi_scale, 'chess_mp') and self.multi_scale.chess_mp is not None:
            for p in self.multi_scale.chess_mp.parameters():
                p.requires_grad = False

    def _unfreeze_csmp_params(self):
        if hasattr(self.multi_scale, 'chess_mp') and self.multi_scale.chess_mp is not None:
            for p in self.multi_scale.chess_mp.parameters():
                p.requires_grad = True

    def _freeze_perceiver_params(self):
        for p in self.perceiver.parameters():
            p.requires_grad = False
        if hasattr(self.multi_scale, 'side_token'):
            for p in self.multi_scale.side_token.parameters():
                p.requires_grad = False

    def _unfreeze_perceiver_params(self):
        for p in self.perceiver.parameters():
            p.requires_grad = True
        if hasattr(self.multi_scale, 'side_token'):
            for p in self.multi_scale.side_token.parameters():
                p.requires_grad = True

    def _freeze_xattn_params(self):
        for xattn in self.gated_xattns:
            for p in xattn.parameters():
                p.requires_grad = False
        if self.shared_readout is not None:
            for p in self.shared_readout.parameters():
                p.requires_grad = False

    def _unfreeze_xattn_params(self):
        for xattn in self.gated_xattns:
            for p in xattn.parameters():
                p.requires_grad = True
        if self.shared_readout is not None:
            for p in self.shared_readout.parameters():
                p.requires_grad = True

    def _freeze_prepend_latent_params(self):
        if self.prepend_latent_readout is not None:
            for p in self.prepend_latent_readout.parameters():
                p.requires_grad = False

    def _unfreeze_prepend_latent_params(self):
        if self.prepend_latent_readout is not None:
            for p in self.prepend_latent_readout.parameters():
                p.requires_grad = True

    def _freeze_lm_pseudotoken_params(self):
        for pseudo in self.lm_pseudotoken_layers:
            for p in pseudo.parameters():
                p.requires_grad = False

    def _unfreeze_lm_pseudotoken_params(self):
        for pseudo in self.lm_pseudotoken_layers:
            for p in pseudo.parameters():
                p.requires_grad = True

    # â”€â”€ LLM Layer Injection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _find_decoder_layers(llm_model: nn.Module):
        """Locate the decoder layer list for different HuggingFace model architectures.
        
        Supports LLaMA/Mistral/Qwen (.model.layers), GPT-NeoX (.gpt_neox.layers),
        GPT-2 (.transformer.h), and PEFT-wrapped variants.
        
        Returns:
            nn.ModuleList of decoder layers, or None if not found.
        """
        # Unwrap PEFT to get the underlying HF model
        base = llm_model
        if hasattr(llm_model, 'base_model'):
            base = llm_model.base_model
            if hasattr(base, 'model'):
                base = base.model

        # LLaMA / Mistral / Qwen / Phi: model.model.layers
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            return base.model.layers
        # GPT-NeoX (ChessGPT): model.gpt_neox.layers
        if hasattr(base, 'gpt_neox') and hasattr(base.gpt_neox, 'layers'):
            return base.gpt_neox.layers
        # GPT-2: model.transformer.h
        if hasattr(base, 'transformer') and hasattr(base.transformer, 'h'):
            return base.transformer.h
        # Fallback: direct .layers attribute
        if hasattr(base, 'layers'):
            return base.layers
        return None

    def inject_into_llm(self, llm_model: nn.Module):
        """
        Wrap specified LLM decoder layers with FusionDecoderLayer.
        Call this ONCE after constructing the LLM.

        Args:
            llm_model: HuggingFace causal LM (e.g., TinyLlama, ChessGPT/GPT-NeoX, GPT-2)
        """
        # Find the decoder layers â€” supports multiple HF architectures
        layers = self._find_decoder_layers(llm_model)
        if layers is None:
            raise RuntimeError(
                f"Cannot locate decoder layers in model: {type(llm_model)}. "
                f"Supported architectures: LLaMA/Mistral (.model.layers), "
                f"GPT-NeoX (.gpt_neox.layers), GPT-2 (.transformer.h)"
            )

        total_layers = len(layers)

        # Wire shared readout into each GatedCrossAttention and set layer fraction
        xattn_by_layer: Dict[int, GatedCrossAttention] = {}
        for i, xattn in zip(self.xattn_layer_indices, self.gated_xattns):
            if i >= total_layers:
                print(f"  WARNING: xattn layer index {i} >= num layers {total_layers}, skipping")
                continue
            xattn.tau = i / total_layers
            if self.xattn_mode == 'recurrent_query_attn':
                xattn._shared_readout = self.shared_readout
            else:
                xattn._shared_readout = None
            xattn_by_layer[i] = xattn

        pseudotoken_by_layer: Dict[int, LayerPseudotokenAttention] = {}
        for i, pseudo in zip(self.pseudotoken_layer_indices, self.lm_pseudotoken_layers):
            if i >= total_layers:
                print(f"  WARNING: pseudotoken layer index {i} >= num layers {total_layers}, skipping")
                continue
            pseudotoken_by_layer[i] = pseudo

        self._fusion_layers = []
        for i in range(total_layers):
            xattn = xattn_by_layer.get(i)
            pseudo = pseudotoken_by_layer.get(i)
            if xattn is None and pseudo is None:
                continue

            original = layers[i]
            wrapper = FusionDecoderLayer(original, xattn, pseudo)
            layers[i] = wrapper
            self._fusion_layers.append(wrapper)

            injected_parts: List[str] = []
            if xattn is not None:
                injected_parts.append(f"GatedCrossAttention (tau={xattn.tau:.3f})")
            if pseudo is not None:
                injected_parts.append(f"LM pseudotokens ({pseudo.num_tokens})")
            print(f"  Injected {' + '.join(injected_parts)} at LLM layer {i}")

    def set_chess_context(
        self,
        perceiver_latents: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        csmp_square_tokens: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        policy_latents: Optional[torch.Tensor] = None,
    ):
        """Set Perceiver latents (and optionally pre-Perceiver context) on all fusion decoder layers."""
        for layer in self._fusion_layers:
            layer.set_chess_context(
                perceiver_latents,
                context,
                csmp_square_tokens=csmp_square_tokens,
                text_attention_mask=text_attention_mask,
                policy_latents=policy_latents,
            )

    def clear_chess_context(self):
        """Clear chess context from all fusion decoder layers after LLM forward."""
        for layer in self._fusion_layers:
            layer.clear_chess_context()

    def set_last_token_trace_capture(self, enabled: bool) -> None:
        for xattn in self.gated_xattns:
            xattn.set_last_token_trace_capture(enabled)

    def clear_last_token_traces(self) -> None:
        for xattn in self.gated_xattns:
            xattn.clear_last_token_trace()

    # â”€â”€ Main Interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def get_num_prefix_tokens(self) -> int:
        """Number of tokens prepended to LLM input."""
        if self.enable_lm_prepend_latents:
            return self.num_lm_prepend_latents
        return 0

    @_compiler_disable
    def get_policy_logits_from_backbone(self,
                                         elo_self: torch.Tensor = None,
                                         elo_oppo: torch.Tensor = None) -> torch.Tensor:
        """Get student backbone policy logits using cached CNN output.

        Uses the CNN output cached by MultiScaleFeatureExtractor's hook,
        then runs only patch_embed -> transformer -> fc_1 under no_grad.
        This avoids re-running the CNN which was already executed for
        multi-scale feature extraction.
        """
        cached_cnn = self.multi_scale._cached_cnn_final
        if cached_cnn is None:
            raise RuntimeError(
                "No cached CNN output â€” call multi_scale.forward() first"
            )
        with torch.no_grad():
            logits = self.backbone.get_policy_from_cnn_output(
                cached_cnn.detach(), elo_self, elo_oppo
            )
        return logits.detach()

    @_compiler_disable
    def get_policy_logits_from_teacher(
        self,
        boards: torch.Tensor,
        elo_self: torch.Tensor = None,
        elo_oppo: torch.Tensor = None,
    ) -> Optional[torch.Tensor]:
        """Get objective policy targets from the frozen Maia teacher."""
        if self.teacher_backbone is None:
            return None
        with torch.no_grad():
            logits = self.teacher_backbone(boards, elo_self=elo_self, elo_oppo=elo_oppo)
        return logits.detach()

    @torch.no_grad()
    def get_gate_values(self) -> Dict[str, float]:
        """
        Extract effective gate values (after tanh) for all gated cross-attention layers.
        
        Returns:
            Dict mapping parameter names to float values.
        """
        gate_values = {}
        for i, idx in enumerate(self.xattn_layer_indices):
            xattn = self.gated_xattns[i]
            
            # 1. Per-head attention gates
            # gate shape is (1, n_heads, 1, 1)
            attn_gates = torch.tanh(xattn.gate).squeeze().cpu().tolist()
            if isinstance(attn_gates, float):  # Case where n_heads=1
                attn_gates = [attn_gates]
                
            for h, val in enumerate(attn_gates):
                gate_values[f"gates/layer_{idx}_head_{h}"] = val
            
            # 2. Scalar FFN gate
            ffn_gate = torch.tanh(xattn.ffn_gate).item()
            gate_values[f"gates/layer_{idx}_ffn"] = ffn_gate

            if xattn.xattn_mode == "recurrent_query_attn":
                gate_values[f"recurrent_query/layer_{idx}_gate_perceiver"] = torch.tanh(xattn.source_gate_perc).item()
                gate_values[f"recurrent_query/layer_{idx}_gate_csmp"] = torch.tanh(xattn.source_gate_csmp).item()
                gate_values[f"recurrent_query/layer_{idx}_gate_policy"] = torch.tanh(xattn.source_gate_policy).item()
            if xattn._last_recurrent_query_state is not None:
                rq = xattn._last_recurrent_query_state
                gate_values[f"recurrent_query/layer_{idx}_state_norm"] = rq.norm(dim=-1).mean().item()
                gate_values[f"recurrent_query/layer_{idx}_state_std"] = rq.std().item()

        return gate_values

    @torch.no_grad()
    def get_structured_xattn_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        source_names = ("csmp", "perceiver", "policy")
        global_source_names = ("perceiver_global", "side_token")

        for i, idx in enumerate(self.xattn_layer_indices):
            xattn = self.gated_xattns[i]
            structured = getattr(xattn, "_last_structured_metrics", None)
            if xattn.xattn_mode != "structured_square_mixer" or structured is None:
                continue

            slot_mean = structured["slot_mean"].detach().float().cpu()
            slot_mean_per_head = structured["slot_mean_per_head"].detach().float().cpu()
            square_mean = structured["square_mean"].detach().float().cpu()
            square_mean_per_head = structured["square_mean_per_head"].detach().float().cpu()
            source_mass = structured["source_mass"].detach().float().cpu()
            source_mass_per_head = structured["source_mass_per_head"].detach().float().cpu()
            global_mean = structured["global_mean"].detach().float().cpu()
            global_mean_per_head = structured["global_mean_per_head"].detach().float().cpu()
            effective_gate_abs_mean_per_head = structured["effective_gate_abs_mean_per_head"].detach().float().cpu()
            token_gate_logit_mean_per_head = structured["token_gate_logit_mean_per_head"].detach().float().cpu()

            metrics[f"structured_xattn/layer_{idx}/router_is_per_head"] = float(
                structured["router_mode"] == "per_head"
            )
            metrics[f"structured_xattn/layer_{idx}/slot_entropy"] = structured["slot_entropy"].item()
            metrics[f"structured_xattn/layer_{idx}/max_slot_mass"] = structured["max_slot_mass"].item()
            metrics[f"structured_xattn/layer_{idx}/max_slot_index"] = structured["max_slot_index"].item()
            metrics[f"structured_xattn/layer_{idx}/max_slot_source_index"] = structured["max_slot_source_index"].item()
            metrics[f"structured_xattn/layer_{idx}/max_slot_square_index"] = structured["max_slot_square_index"].item()
            metrics[f"structured_xattn/layer_{idx}/square_entropy"] = structured["square_entropy"].item()
            metrics[f"structured_xattn/layer_{idx}/square_entropy_norm"] = structured["square_entropy_norm"].item()
            metrics[f"structured_xattn/layer_{idx}/square_usage_entropy"] = structured["square_usage_entropy"].item()
            metrics[f"structured_xattn/layer_{idx}/square_usage_entropy_norm"] = structured["square_usage_entropy_norm"].item()
            metrics[f"structured_xattn/layer_{idx}/max_square_mass"] = structured["max_square_mass"].item()
            metrics[f"structured_xattn/layer_{idx}/max_square_index"] = structured["max_square_index"].item()
            metrics[f"structured_xattn/layer_{idx}/global_entropy"] = structured["global_entropy"].item()
            metrics[f"structured_xattn/layer_{idx}/effective_gate_abs_mean"] = structured["effective_gate_abs_mean"].item()
            metrics[f"structured_xattn/layer_{idx}/effective_gate_std"] = structured["effective_gate_std"].item()
            metrics[f"structured_xattn/layer_{idx}/token_gate_logit_mean"] = structured["token_gate_logit_mean"].item()
            metrics[f"structured_xattn/layer_{idx}/slot_mean_mean"] = slot_mean.mean().item()
            metrics[f"structured_xattn/layer_{idx}/slot_mean_std"] = slot_mean.std(unbiased=False).item()
            metrics[f"structured_xattn/layer_{idx}/slot_mean_min"] = slot_mean.min().item()
            metrics[f"structured_xattn/layer_{idx}/slot_mean_max"] = slot_mean.max().item()
            metrics[f"structured_xattn/layer_{idx}/square_mean_mean"] = square_mean.mean().item()
            metrics[f"structured_xattn/layer_{idx}/square_mean_std"] = square_mean.std(unbiased=False).item()
            metrics[f"structured_xattn/layer_{idx}/square_mean_min"] = square_mean.min().item()
            metrics[f"structured_xattn/layer_{idx}/square_mean_max"] = square_mean.max().item()
            metrics[f"structured_xattn/layer_{idx}/source_mass_mean"] = source_mass.mean().item()
            metrics[f"structured_xattn/layer_{idx}/source_mass_std"] = source_mass.std(unbiased=False).item()
            metrics[f"structured_xattn/layer_{idx}/source_mass_min"] = source_mass.min().item()
            metrics[f"structured_xattn/layer_{idx}/source_mass_max"] = source_mass.max().item()
            metrics[f"structured_xattn/layer_{idx}/global_mass_mean"] = global_mean.mean().item()
            metrics[f"structured_xattn/layer_{idx}/global_mass_std"] = global_mean.std(unbiased=False).item()
            metrics[f"structured_xattn/layer_{idx}/global_mass_min"] = global_mean.min().item()
            metrics[f"structured_xattn/layer_{idx}/global_mass_max"] = global_mean.max().item()

            for source_idx, source_name in enumerate(source_names):
                metrics[f"structured_xattn/layer_{idx}/source_mass/{source_name}"] = source_mass[source_idx].item()
                base = source_idx * 64
                source_slot_mean = slot_mean[base:base + 64]
                source_slot_max_index = int(torch.argmax(source_slot_mean).item())
                source_slot_min_index = int(torch.argmin(source_slot_mean).item())
                metrics[f"structured_xattn/layer_{idx}/{source_name}/slot_mean_mean"] = source_slot_mean.mean().item()
                metrics[f"structured_xattn/layer_{idx}/{source_name}/slot_mean_std"] = source_slot_mean.std(unbiased=False).item()
                metrics[f"structured_xattn/layer_{idx}/{source_name}/slot_mean_min"] = source_slot_mean.min().item()
                metrics[f"structured_xattn/layer_{idx}/{source_name}/slot_mean_max"] = source_slot_mean.max().item()
                metrics[f"structured_xattn/layer_{idx}/{source_name}/slot_min_square_index"] = float(source_slot_min_index)
                metrics[f"structured_xattn/layer_{idx}/{source_name}/slot_max_square_index"] = float(source_slot_max_index)

            for head_idx in range(int(slot_mean_per_head.size(0))):
                metrics[f"structured_xattn/layer_{idx}/head_{head_idx}/slot_mean_std"] = (
                    slot_mean_per_head[head_idx].std(unbiased=False).item()
                )
                metrics[f"structured_xattn/layer_{idx}/head_{head_idx}/square_mean_std"] = (
                    square_mean_per_head[head_idx].std(unbiased=False).item()
                )
                metrics[f"structured_xattn/layer_{idx}/head_{head_idx}/effective_gate_abs_mean"] = (
                    effective_gate_abs_mean_per_head[head_idx].item()
                )
                metrics[f"structured_xattn/layer_{idx}/head_{head_idx}/token_gate_logit_mean"] = (
                    token_gate_logit_mean_per_head[head_idx].item()
                )
                metrics[f"structured_xattn/layer_{idx}/head_{head_idx}/source_mass_mean"] = (
                    source_mass_per_head[head_idx].mean().item()
                )
                metrics[f"structured_xattn/layer_{idx}/head_{head_idx}/global_mass_mean"] = (
                    global_mean_per_head[head_idx].mean().item()
                )

            for global_idx, global_name in enumerate(global_source_names):
                metrics[f"structured_xattn/layer_{idx}/global_mass/{global_name}"] = global_mean[global_idx].item()

        return metrics

    @torch.no_grad()
    def get_last_token_structured_traces(
        self,
        sample_index: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        traces: Dict[int, Dict[str, Any]] = {}

        for i, idx in enumerate(self.xattn_layer_indices):
            xattn = self.gated_xattns[i]
            trace = getattr(xattn, "_last_token_trace", None)
            if xattn.xattn_mode != "structured_square_mixer" or trace is None:
                continue

            batch_size = int(trace["raw_slot_weights"].size(0))
            if sample_index < 0 or sample_index >= batch_size:
                raise IndexError(
                    f"sample_index {sample_index} out of range for trace batch size {batch_size}"
                )

            source_square_weights = (
                trace["source_square_weights"][sample_index].detach().float().cpu()
            )
            trace_dict: Dict[str, Any] = {
                "raw_slot_weights": trace["raw_slot_weights"][sample_index].detach().float().cpu(),
                "source_square_weights": source_square_weights,
                "aggregate_square_weights": (
                    trace["aggregate_square_weights"][sample_index].detach().float().cpu()
                ),
                "global_weights": trace["global_weights"][sample_index].detach().float().cpu(),
                "last_token_index": trace["last_token_indices"][sample_index].detach().cpu(),
                "csmp_square_weights": source_square_weights[0].clone(),
                "perceiver_square_weights": source_square_weights[1].clone(),
                "policy_square_weights": source_square_weights[2].clone(),
            }
            if "effective_head_gates" in trace:
                trace_dict["effective_head_gates"] = (
                    trace["effective_head_gates"][sample_index].detach().float().cpu()
                )
            if "token_gate_logits" in trace:
                trace_dict["token_gate_logits"] = (
                    trace["token_gate_logits"][sample_index].detach().float().cpu()
                )
            if "source_square_contribution_norms" in trace:
                contribution_norms = (
                    trace["source_square_contribution_norms"][sample_index].detach().float().cpu()
                )
                trace_dict["source_square_contribution_norms"] = contribution_norms
                trace_dict["aggregate_square_contribution_norms"] = (
                    trace["aggregate_square_contribution_norms"][sample_index].detach().float().cpu()
                )
                trace_dict["global_contribution_norms"] = (
                    trace["global_contribution_norms"][sample_index].detach().float().cpu()
                )
                trace_dict["csmp_square_contribution_norms"] = contribution_norms[0].clone()
                trace_dict["perceiver_square_contribution_norms"] = contribution_norms[1].clone()
                trace_dict["policy_square_contribution_norms"] = contribution_norms[2].clone()
            if "raw_slot_weights_per_head" in trace:
                per_head_source_square_weights = (
                    trace["source_square_weights_per_head"][sample_index].detach().float().cpu()
                )
                trace_dict["raw_slot_weights_per_head"] = (
                    trace["raw_slot_weights_per_head"][sample_index].detach().float().cpu()
                )
                trace_dict["source_square_weights_per_head"] = per_head_source_square_weights
                trace_dict["aggregate_square_weights_per_head"] = (
                    trace["aggregate_square_weights_per_head"][sample_index].detach().float().cpu()
                )
                trace_dict["global_weights_per_head"] = (
                    trace["global_weights_per_head"][sample_index].detach().float().cpu()
                )
                trace_dict["csmp_square_weights_per_head"] = per_head_source_square_weights[:, 0].clone()
                trace_dict["perceiver_square_weights_per_head"] = per_head_source_square_weights[:, 1].clone()
                trace_dict["policy_square_weights_per_head"] = per_head_source_square_weights[:, 2].clone()
            if "source_square_contribution_norms_per_head" in trace:
                per_head_contribution_norms = (
                    trace["source_square_contribution_norms_per_head"][sample_index].detach().float().cpu()
                )
                trace_dict["source_square_contribution_norms_per_head"] = per_head_contribution_norms
                trace_dict["aggregate_square_contribution_norms_per_head"] = (
                    trace["aggregate_square_contribution_norms_per_head"][sample_index].detach().float().cpu()
                )
                trace_dict["global_contribution_norms_per_head"] = (
                    trace["global_contribution_norms_per_head"][sample_index].detach().float().cpu()
                )
                trace_dict["csmp_square_contribution_norms_per_head"] = (
                    per_head_contribution_norms[:, 0].clone()
                )
                trace_dict["perceiver_square_contribution_norms_per_head"] = (
                    per_head_contribution_norms[:, 1].clone()
                )
                trace_dict["policy_square_contribution_norms_per_head"] = (
                    per_head_contribution_norms[:, 2].clone()
                )
            trace_dict["router_mode"] = trace["router_mode"]
            traces[idx] = trace_dict

        return traces

    def compute_structured_xattn_sparse_loss(self, device: torch.device) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for xattn in self.gated_xattns:
            if xattn.xattn_mode != "structured_square_mixer":
                continue
            loss = getattr(xattn, "_last_structured_square_sparse_loss", None)
            if loss is not None:
                losses.append(loss)
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=device)

    def compute_structured_xattn_square_diversity_loss(
        self,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        losses: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        target_entropy = float(getattr(self.cfg, "structured_xattn_square_diversity_target_entropy", 0.5))
        target_entropy = min(max(target_entropy, 0.0), 1.0)
        for xattn in self.gated_xattns:
            if xattn.xattn_mode != "structured_square_mixer":
                continue
            usage_entropy = getattr(xattn, "_last_structured_square_usage_entropy_norm", None)
            if usage_entropy is None:
                continue
            target = usage_entropy.new_tensor(target_entropy)
            entropies.append(usage_entropy)
            losses.append(F.relu(target - usage_entropy))
        if losses:
            return torch.stack(losses).mean(), torch.stack(entropies).mean()
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    def compute_structured_xattn_gate_usage_loss(
        self,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        losses: List[torch.Tensor] = []
        usages: List[torch.Tensor] = []
        target_usage = float(getattr(self.cfg, "structured_xattn_gate_usage_target", 0.1))
        target_usage = min(max(target_usage, 0.0), 1.0)
        for xattn in self.gated_xattns:
            if xattn.xattn_mode != "structured_square_mixer":
                continue
            usage = getattr(xattn, "_last_structured_gate_usage_mean_abs", None)
            if usage is None:
                continue
            target = usage.new_tensor(target_usage)
            usages.append(usage)
            losses.append(F.relu(target - usage))
        if losses:
            return torch.stack(losses).mean(), torch.stack(usages).mean()
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    @torch.no_grad()
    def revive_recurrent_text_state_if_stuck(self, scale_value: float = 1.0) -> int:
        """
        Backward-compatibility helper for checkpoints created with a dead init:
        recurrent readout projection all-zeros AND recurrent scale exactly zero.

        Returns:
            1 if revived, else 0.
        """
        # Legacy no-op kept for backward compatibility with training loop hooks.
        return 0

    def forward(
        self,
        boards: torch.Tensor,
        elo_self: torch.Tensor = None,
        elo_oppo: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            boards: (B, 18, 8, 8)
            elo_self: (B,) or None
            elo_oppo: (B,) or None
            **kwargs: side_to_move, engineered_features

        Returns:
            dict with keys:
                perceiver_latents: (B, num_latents, perceiver_dim)
                csmp_square_tokens: (B, 64, tap_dim) or None
                policy_logits: (B, 1880)
                eval_logits: (B, num_eval_buckets)
                policy_targets: (B, 1880) detached Maia policy for distillation
                move_eval_logits: (B, 1880) CE/pairwise branch logits
                move_eval_mse_logits: (B, 1880) MSE branch logits
        """
        batch_size = boards.size(0)
        device = boards.device
        _profile = getattr(self, '_profile', False)

        if _profile:
            import time
            torch.cuda.synchronize(device)
            _t0 = time.perf_counter()

        # Side-to-move tensor
        side_to_move = kwargs.get('side_to_move', None)
        if side_to_move is None:
            side_tensor = torch.ones(batch_size, dtype=torch.long, device=device)
        elif isinstance(side_to_move, torch.Tensor):
            side_tensor = side_to_move.long().to(device)
        else:
            side_tensor = torch.tensor(
                [1 if s else 0 for s in side_to_move],
                dtype=torch.long, device=device
            )

        # Compute absolute-space board tensor for CSMP, BSR, SPP
        # (un-mirrors spatial flip + piece color swap for Black-to-move samples)
        if getattr(self.cfg, 'use_absolute_coords', True):
            abs_boards = unmirror_board_tensor(boards, side_tensor)
        else:
            abs_boards = None  # legacy: pass perspective-relative boards through

        # Multi-scale feature extraction (includes spatial pos-enc + side token)
        # Propagate _profile flag to CSMP for granular timing when enabled
        if hasattr(self.multi_scale, 'chess_mp') and self.multi_scale.chess_mp is not None:
            self.multi_scale.chess_mp._profile = _profile
        multi_scale_out = self.multi_scale(
            boards, elo_self, elo_oppo,
            side_to_move=side_tensor, abs_boards=abs_boards,
            return_components=True,
        )
        context = multi_scale_out["context"]
        csmp_square_tokens = multi_scale_out.get("csmp_square_tokens", None)

        if _profile:
            torch.cuda.synchronize(device)
            _t1 = time.perf_counter()

        # Perceiver â€” returns shared latents (each xattn layer reads them independently)
        engineered_features = kwargs.get('engineered_features', None)
        self.perceiver._profile = bool(_profile)
        (
            latents,
            aux_repr,
            policy_logits,
            eval_logits,
            policy_latents,
            entropy_metrics,
            move_eval_logits,
            move_eval_mse_logits,
            move_mate_logits,
        ) = self.perceiver(
            context,
            engineered_features,
            side_to_move=side_tensor,
            latents_are_absolute=(abs_boards is not None),
        )

        if _profile:
            torch.cuda.synchronize(device)
            _t2 = time.perf_counter()

        # Policy distillation targets: prefer precomputed cache, fall back to live teacher
        precomputed_policy = kwargs.get('precomputed_policy', None)
        if getattr(self.cfg, 'use_precomputed_policy', False) and precomputed_policy is not None:
            policy_targets = precomputed_policy.to(device)
        elif self.use_cnn:
            policy_targets = self.get_policy_logits_from_teacher(boards, elo_self, elo_oppo)
        else:
            policy_targets = None  # no backbone to distill from

        if _profile:
            torch.cuda.synchronize(device)
            _t3 = time.perf_counter()
            _csmp_ms = 0.0
            if hasattr(self.multi_scale, 'chess_mp') and self.multi_scale.chess_mp is not None:
                _csmp_ms = getattr(self.multi_scale.chess_mp, '_last_ms', 0.0)
            self._last_profile_ms = {
                "multi_scale":    (_t1 - _t0) * 1000,
                "csmp":           _csmp_ms,
                "perceiver":      (_t2 - _t1) * 1000,
                "policy_head":    float(entropy_metrics.get("policy_head_ms", torch.tensor(0.0, device=device)).item()) if isinstance(entropy_metrics, dict) else 0.0,
                "policy_targets": (_t3 - _t2) * 1000,
                "total":          (_t3 - _t0) * 1000,
            }
            print(f"  [PROFILE adapter] multi_scale={self._last_profile_ms['multi_scale']:.1f}ms  "
                  f"perceiver={self._last_profile_ms['perceiver']:.1f}ms  "
                  f"policy_head={self._last_profile_ms['policy_head']:.1f}ms  "
                  f"policy_targets={self._last_profile_ms['policy_targets']:.1f}ms  "
                  f"total={self._last_profile_ms['total']:.1f}ms")

        prepend_embeddings = None
        if self.prepend_latent_readout is not None:
            prepend_embeddings = self.prepend_latent_readout(
                perceiver_latents=latents,
                csmp_square_tokens=csmp_square_tokens,
                policy_latents=policy_latents,
            )

        result = {
            'perceiver_latents': latents,
            'csmp_square_tokens': csmp_square_tokens,
            'prepend_embeddings': prepend_embeddings,
            'policy_logits': policy_logits,
            'eval_logits': eval_logits,
            'policy_latents': policy_latents,
            'policy_targets': policy_targets,
            'entropy_metrics': entropy_metrics,
            'move_eval_logits': move_eval_logits,
            'move_eval_mse_logits': move_eval_mse_logits,
            'move_mate_logits': move_mate_logits,
        }

        # --- BSR / SPP auxiliary heads (run only when weight > 0) ---
        target_boards = abs_boards if abs_boards is not None else boards
        if _profile:
            torch.cuda.synchronize(device)
            _t4 = time.perf_counter()

        if getattr(self.cfg, 'bsr_weight', 0.0) > 0:
            result['bsr_logits'] = self.bsr_head(latents)  # (B, 64, 13)
            result['bsr_targets'] = extract_piece_types(target_boards)  # (B, 64)

        if getattr(self.cfg, 'spp_weight', 0.0) > 0:
            result['spp_preds'] = self.spp_head(latents)  # (B, 64, 10)
            result['spp_targets'] = compute_spp_targets(target_boards, self.spp_mask_builder)  # (B, 64, 10)

        if _profile:
            torch.cuda.synchronize(device)
            _t5 = time.perf_counter()
            self._last_profile_ms["bsr_spp"] = (_t5 - _t4) * 1000
            self._last_profile_ms["total"] = (_t5 - _t0) * 1000
            print(f"  [PROFILE adapter] bsr_spp={self._last_profile_ms['bsr_spp']:.1f}ms  "
                  f"(updated total={self._last_profile_ms['total']:.1f}ms)")

        # Pass pre-Perceiver CSMP context for shared readout
        result['context'] = context

        return result

    def compute_auxiliary_losses(
        self,
        policy_logits: torch.Tensor,
        eval_logits: Optional[torch.Tensor],
        policy_targets: torch.Tensor,
        policy_mask: Optional[torch.Tensor] = None,
        eval_targets: Optional[torch.Tensor] = None,
        bsr_logits: Optional[torch.Tensor] = None,
        bsr_targets: Optional[torch.Tensor] = None,
        spp_preds: Optional[torch.Tensor] = None,
        spp_targets: Optional[torch.Tensor] = None,
        move_eval_logits: Optional[torch.Tensor] = None,
        move_eval_mse_logits: Optional[torch.Tensor] = None,
        move_mate_logits: Optional[torch.Tensor] = None,
        move_eval_indices: Optional[torch.Tensor] = None,
        move_eval_targets: Optional[torch.Tensor] = None,
        move_eval_mask: Optional[torch.Tensor] = None,
        move_ce_indices: Optional[torch.Tensor] = None,
        move_ce_targets: Optional[torch.Tensor] = None,
        move_ce_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses.

        Args:
            policy_logits: (B, 1880) from perceiver
            eval_logits: (B, num_eval_buckets) from perceiver, or None if disabled
            policy_targets: (B, 1880) from Maia backbone (detached)
            policy_mask: (B,) bool mask; True where policy_targets are valid.
            eval_targets: (B,) long tensor, class indices. None = skip eval loss.
            bsr_logits: (B, 64, 13) piece-type logits per square. None = skip BSR.
            bsr_targets: (B, 64) long, piece-type indices 0-12. None = skip BSR.
            spp_preds: (B, 64, 10) attack counts + ray distances. None = skip SPP.
            spp_targets: (B, 64, 10) ground truth. None = skip SPP.
            move_eval_logits: (B, 1880) CE/pairwise branch move scores.
            move_eval_mse_logits: (B, 1880) MSE branch move scores.
            move_mate_logits: (B, 1880) per-move mate logits. None = skip mate BCE.
            move_eval_indices: (B, M) vocab indices for supervised moves.
            move_eval_targets: (B, M) centipawn values (float) for supervised moves.
            move_eval_mask: (B, M) bool mask for valid entries.
            move_ce_indices: (B, K) CE-candidate move indices (prefer Stockfish top-k).
            move_ce_targets: (B, K) CE-candidate target cp values.
            move_ce_mask: (B, K) bool mask for valid CE candidates.

        Returns:
            dict with 'policy_loss', 'eval_loss', 'bsr_loss', 'spp_loss',
            'move_eval_loss', 'move_eval_mse', 'move_eval_ce', 'move_eval_pairwise',
            'move_eval_mate', 'structured_xattn_sparse_loss',
            'structured_xattn_square_diversity_loss',
            'structured_xattn_square_usage_entropy',
            'structured_xattn_gate_usage_loss',
            'structured_xattn_gate_usage_mean_abs', 'total_aux_loss'
        """
        losses = {}

        # Policy distillation: KL divergence
        if self.cfg.aux_policy_weight > 0 and policy_targets is not None:
            if policy_mask is not None:
                valid = policy_mask.to(device=policy_logits.device, dtype=torch.bool).view(-1)
            else:
                valid = torch.ones(policy_logits.shape[0], dtype=torch.bool, device=policy_logits.device)
            if valid.any():
                log_probs = F.log_softmax(policy_logits[valid], dim=-1)
                target_probs = F.softmax(policy_targets[valid], dim=-1)
                policy_loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
                losses['policy_loss'] = policy_loss
            else:
                losses['policy_loss'] = torch.tensor(0.0, device=policy_logits.device)
        else:
            losses['policy_loss'] = torch.tensor(0.0, device=policy_logits.device)

        # Eval classification (optional/deprecated in simplified eval objective)
        if self.cfg.aux_eval_weight > 0 and eval_targets is not None and eval_logits is not None:
            eval_loss = F.cross_entropy(eval_logits, eval_targets)
            losses['eval_loss'] = eval_loss
        else:
            losses['eval_loss'] = torch.tensor(0.0, device=policy_logits.device)

        # BSR: Board State Reconstruction (cross-entropy over 13 piece classes Ã— 64 squares)
        if getattr(self.cfg, 'bsr_weight', 0.0) > 0 and bsr_logits is not None and bsr_targets is not None:
            bsr_loss = F.cross_entropy(
                bsr_logits.reshape(-1, 13),  # (B*64, 13)
                bsr_targets.reshape(-1),     # (B*64,)
            )
            losses['bsr_loss'] = bsr_loss
        else:
            losses['bsr_loss'] = torch.tensor(0.0, device=policy_logits.device)

        # SPP: Square Property Prediction (smooth L1 over 10 channels Ã— 64 squares)
        if getattr(self.cfg, 'spp_weight', 0.0) > 0 and spp_preds is not None and spp_targets is not None:
            spp_loss = F.smooth_l1_loss(spp_preds, spp_targets)
            losses['spp_loss'] = spp_loss
        else:
            losses['spp_loss'] = torch.tensor(0.0, device=policy_logits.device)

        # Per-move evaluation: mixed objective (MSE + soft CE + pairwise ranking + mate BCE)
        # move_eval_targets: (B, M) centipawn values (float), NOT bucket indices
        move_eval_weight = getattr(self.cfg, 'aux_move_eval_weight', 0.0)
        if (move_eval_weight > 0
                and move_eval_indices is not None
                and move_eval_targets is not None
                and move_eval_mask is not None):
            rank_logits = move_eval_logits
            mse_logits = move_eval_mse_logits if move_eval_mse_logits is not None else move_eval_logits

            cp_scale = getattr(self.cfg, 'move_eval_cp_scale', 512.0)
            cp_clip = float(max(1.0, getattr(self.cfg, 'move_eval_cp_clip', 2000.0)))
            mate_thresh = float(max(cp_clip, getattr(self.cfg, 'move_eval_mate_threshold_cp', 9000.0)))
            mse_w = getattr(self.cfg, 'move_eval_mse_weight', 0.5)
            ce_w = getattr(self.cfg, 'move_eval_ce_weight', 0.5)
            pairwise_w = getattr(self.cfg, 'move_eval_pairwise_weight', 0.0)
            mate_w = getattr(self.cfg, 'move_eval_mate_weight', 0.25)
            ce_topk = int(max(1, getattr(self.cfg, 'move_eval_ce_topk', 5)))
            ce_temp = float(max(1e-6, getattr(self.cfg, 'move_eval_ce_cp_temperature', 128.0)))
            pairwise_topk = int(max(2, getattr(self.cfg, 'move_eval_pairwise_topk', 5)))
            pairwise_margin = float(max(0.0, getattr(self.cfg, 'move_eval_pairwise_cp_margin', 0.0)))
            pairwise_temp = float(max(1e-6, getattr(self.cfg, 'move_eval_pairwise_temperature', 1.0)))

            # --- MSE regression: predict clipped normalized centipawn eval per move ---
            clipped_targets_cp = move_eval_targets.clamp(min=-cp_clip, max=cp_clip)
            normalized_targets = clipped_targets_cp / cp_scale
            if mse_logits is not None:
                pred_scores = mse_logits.gather(
                    dim=1, index=move_eval_indices,
                )  # (B, M) raw eval scores
                masked_pred = pred_scores[move_eval_mask]       # (N,)
                masked_tgt = normalized_targets[move_eval_mask]  # (N,)
                if masked_tgt.numel() > 0:
                    mse_loss = F.mse_loss(masked_pred, masked_tgt)
                else:
                    mse_loss = torch.tensor(0.0, device=policy_logits.device)
            else:
                mse_loss = torch.tensor(0.0, device=policy_logits.device)

            # --- Mate BCE: classify whether move leads to mate (win/loss mate treated as mate=1) ---
            if move_mate_logits is not None:
                mate_pred_scores = move_mate_logits.gather(dim=1, index=move_eval_indices)  # (B, M)
                mate_targets = (move_eval_targets.abs() >= mate_thresh).to(dtype=mate_pred_scores.dtype)
                masked_mate_pred = mate_pred_scores[move_eval_mask]
                masked_mate_tgt = mate_targets[move_eval_mask]
                if masked_mate_tgt.numel() > 0:
                    mate_loss = F.binary_cross_entropy_with_logits(masked_mate_pred, masked_mate_tgt)
                else:
                    mate_loss = torch.tensor(0.0, device=policy_logits.device)
            else:
                mate_loss = torch.tensor(0.0, device=policy_logits.device)

            # --- CE on move-eval logits with soft Stockfish targets (top-k supervised moves) ---
            # For each position:
            #   1) Select top-k moves by target cp from available supervised set.
            #   2) Build soft targets via softmax(cp / temperature).
            #   3) Compute sparse CE against full move-eval logits (all other moves target prob = 0).
            ce_indices = move_ce_indices if move_ce_indices is not None else move_eval_indices
            ce_targets_cp = move_ce_targets if move_ce_targets is not None else move_eval_targets
            ce_mask = move_ce_mask if move_ce_mask is not None else move_eval_mask
            valid_positions = ce_mask.any(dim=1)
            if valid_positions.any() and ce_w > 0 and rank_logits is not None:
                row_losses = []
                row_log_probs = F.log_softmax(rank_logits[valid_positions], dim=-1)  # (B', 1880)
                row_idx = ce_indices[valid_positions]    # (B', K)
                row_cp = ce_targets_cp[valid_positions]  # (B', K)
                row_mask = ce_mask[valid_positions]      # (B', K)

                for r in range(row_log_probs.size(0)):
                    m = row_mask[r]
                    if not m.any():
                        continue
                    cand_idx = row_idx[r][m]      # (N,)
                    cand_cp = row_cp[r][m].clamp(min=-cp_clip, max=cp_clip)  # (N,)
                    k = min(ce_topk, int(cand_idx.numel()))
                    if k <= 0:
                        continue
                    top_cp, top_pos = torch.topk(cand_cp, k=k, largest=True, sorted=False)
                    top_idx = cand_idx[top_pos]   # (k,)
                    target_probs = F.softmax(top_cp / ce_temp, dim=0)  # (k,)
                    logp = row_log_probs[r].gather(0, top_idx)          # (k,)
                    row_losses.append(-(target_probs * logp).sum())

                if row_losses:
                    ce_loss = torch.stack(row_losses).mean()
                else:
                    ce_loss = torch.tensor(0.0, device=policy_logits.device)
            else:
                ce_loss = torch.tensor(0.0, device=policy_logits.device)

            # --- Pairwise ranking on move-eval logits (top-k supervised moves) ---
            # For each position:
            #   1) Select top-k moves by target cp.
            #   2) Build ordered pairs where cp_i > cp_j (+ optional margin).
            #   3) Penalize inverted ordering with logistic loss softplus(-(s_i - s_j)).
            if valid_positions.any() and pairwise_w > 0 and rank_logits is not None:
                row_losses = []
                row_scores = rank_logits[valid_positions]  # (B', 1880)
                row_idx = ce_indices[valid_positions]    # (B', K)
                row_cp = ce_targets_cp[valid_positions]  # (B', K)
                row_mask = ce_mask[valid_positions]      # (B', K)

                for r in range(row_scores.size(0)):
                    m = row_mask[r]
                    if int(m.sum().item()) < 2:
                        continue
                    cand_idx = row_idx[r][m]      # (N,)
                    cand_cp = row_cp[r][m].clamp(min=-cp_clip, max=cp_clip)  # (N,)
                    k = min(pairwise_topk, int(cand_idx.numel()))
                    if k < 2:
                        continue
                    top_cp, top_pos = torch.topk(cand_cp, k=k, largest=True, sorted=False)
                    top_idx = cand_idx[top_pos]  # (k,)
                    top_scores = row_scores[r].gather(0, top_idx)  # (k,)
                    cp_diff = top_cp.unsqueeze(1) - top_cp.unsqueeze(0)  # (k, k)
                    pair_mask = cp_diff > pairwise_margin
                    if not pair_mask.any():
                        continue
                    score_diff = (top_scores.unsqueeze(1) - top_scores.unsqueeze(0)) / pairwise_temp
                    row_losses.append(F.softplus(-score_diff[pair_mask]).mean())

                if row_losses:
                    pairwise_loss = torch.stack(row_losses).mean()
                else:
                    pairwise_loss = torch.tensor(0.0, device=policy_logits.device)
            else:
                pairwise_loss = torch.tensor(0.0, device=policy_logits.device)

            losses['move_eval_mse'] = mse_loss
            losses['move_eval_ce'] = ce_loss
            losses['move_eval_pairwise'] = pairwise_loss
            losses['move_eval_mate'] = mate_loss
            losses['move_eval_loss'] = (
                mse_w * mse_loss
                + ce_w * ce_loss
                + pairwise_w * pairwise_loss
                + mate_w * mate_loss
            )
        else:
            losses['move_eval_loss'] = torch.tensor(0.0, device=policy_logits.device)
            losses['move_eval_mse'] = torch.tensor(0.0, device=policy_logits.device)
            losses['move_eval_ce'] = torch.tensor(0.0, device=policy_logits.device)
            losses['move_eval_pairwise'] = torch.tensor(0.0, device=policy_logits.device)
            losses['move_eval_mate'] = torch.tensor(0.0, device=policy_logits.device)

        structured_xattn_sparse_weight = getattr(self.cfg, 'structured_xattn_sparse_weight', 0.0)
        if structured_xattn_sparse_weight > 0:
            losses['structured_xattn_sparse_loss'] = self.compute_structured_xattn_sparse_loss(
                device=policy_logits.device,
            )
        else:
            losses['structured_xattn_sparse_loss'] = torch.tensor(0.0, device=policy_logits.device)
        structured_xattn_square_diversity_weight = getattr(
            self.cfg,
            'structured_xattn_square_diversity_weight',
            0.0,
        )
        if structured_xattn_square_diversity_weight > 0:
            (
                losses['structured_xattn_square_diversity_loss'],
                losses['structured_xattn_square_usage_entropy'],
            ) = self.compute_structured_xattn_square_diversity_loss(
                device=policy_logits.device,
            )
        else:
            losses['structured_xattn_square_diversity_loss'] = torch.tensor(0.0, device=policy_logits.device)
            losses['structured_xattn_square_usage_entropy'] = torch.tensor(0.0, device=policy_logits.device)
        structured_xattn_gate_usage_weight = getattr(
            self.cfg,
            'structured_xattn_gate_usage_weight',
            0.0,
        )
        if structured_xattn_gate_usage_weight > 0:
            (
                losses['structured_xattn_gate_usage_loss'],
                losses['structured_xattn_gate_usage_mean_abs'],
            ) = self.compute_structured_xattn_gate_usage_loss(
                device=policy_logits.device,
            )
        else:
            losses['structured_xattn_gate_usage_loss'] = torch.tensor(0.0, device=policy_logits.device)
            losses['structured_xattn_gate_usage_mean_abs'] = torch.tensor(0.0, device=policy_logits.device)

        losses['total_aux_loss'] = (
            self.cfg.aux_policy_weight * losses['policy_loss']
            + self.cfg.aux_eval_weight * losses['eval_loss']
            + getattr(self.cfg, 'bsr_weight', 0.0) * losses['bsr_loss']
            + getattr(self.cfg, 'spp_weight', 0.0) * losses['spp_loss']
            + move_eval_weight * losses['move_eval_loss']
            + structured_xattn_sparse_weight * losses['structured_xattn_sparse_loss']
            + structured_xattn_square_diversity_weight * losses['structured_xattn_square_diversity_loss']
            + structured_xattn_gate_usage_weight * losses['structured_xattn_gate_usage_loss']
        )

        return losses




