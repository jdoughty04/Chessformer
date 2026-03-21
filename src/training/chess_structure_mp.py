"""
Chess Structure Message Passing (CSMP)

Multi-relation sparse attention over 64 board-square tokens.
Each attention head is assigned to a specific chess-topological relation
and masked so that each square only attends to structurally related squares.

Heads 0-5: static topology (file, rank, diagonal, anti-diagonal, knight, king)
Head 6: dynamic ray mask (unobstructed sliding path based on current occupancy)
Head 7: dynamic attack mask (piece on i attacks j, accounts for piece type + obstructions)
Heads 8+: global attention (all squares visible)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Piece-type extraction from board tensor
# =============================================================================

def extract_piece_types(boards: torch.Tensor) -> torch.Tensor:
    """
    Extract per-square piece type indices from Maia2 board tensor.

    Board tensor channels 0-11:
        0-5:  White pieces (P, N, B, R, Q, K)
        6-11: Black pieces (P, N, B, R, Q, K)

    Returns:
        (B, 64) long tensor with values 0-12:
            0=wP, 1=wN, 2=wB, 3=wR, 4=wQ, 5=wK,
            6=bP, 7=bN, 8=bB, 9=bR, 10=bQ, 11=bK,
            12=empty
    """
    piece_planes = boards[:, :12, :, :]  # (B, 12, 8, 8)
    occupancy = piece_planes.sum(dim=1)  # (B, 8, 8)
    piece_type = piece_planes.argmax(dim=1)  # (B, 8, 8) — arbitrary for empty squares
    piece_type = torch.where(
        occupancy > 0.5, piece_type, torch.full_like(piece_type, 12)
    )
    return piece_type.reshape(boards.size(0), 64)  # (B, 64)


# =============================================================================
# Static topology masks
# =============================================================================

def _sq_to_rf(sq: int) -> Tuple[int, int]:
    """Square index (0-63) to (rank, file)."""
    return sq // 8, sq % 8


def build_file_mask() -> torch.Tensor:
    """(64, 64) bool — True if same file."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for i in range(64):
        ri, fi = _sq_to_rf(i)
        for j in range(64):
            rj, fj = _sq_to_rf(j)
            if fi == fj:
                mask[i, j] = True
    return mask


def build_rank_mask() -> torch.Tensor:
    """(64, 64) bool — True if same rank."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for i in range(64):
        ri, fi = _sq_to_rf(i)
        for j in range(64):
            rj, fj = _sq_to_rf(j)
            if ri == rj:
                mask[i, j] = True
    return mask


def build_diagonal_mask() -> torch.Tensor:
    """(64, 64) bool — True if same NW-SE diagonal (rank - file = constant)."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for i in range(64):
        ri, fi = _sq_to_rf(i)
        for j in range(64):
            rj, fj = _sq_to_rf(j)
            if (ri - fi) == (rj - fj):
                mask[i, j] = True
    return mask


def build_anti_diagonal_mask() -> torch.Tensor:
    """(64, 64) bool — True if same NE-SW anti-diagonal (rank + file = constant)."""
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for i in range(64):
        ri, fi = _sq_to_rf(i)
        for j in range(64):
            rj, fj = _sq_to_rf(j)
            if (ri + fi) == (rj + fj):
                mask[i, j] = True
    return mask


def build_knight_mask() -> torch.Tensor:
    """(64, 64) bool — True if knight-move apart (+ self-connection)."""
    deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2), (1, 2), (2, -1), (2, 1)]
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for i in range(64):
        ri, fi = _sq_to_rf(i)
        mask[i, i] = True  # self-connection
        for dr, df in deltas:
            rj, fj = ri + dr, fi + df
            if 0 <= rj < 8 and 0 <= fj < 8:
                j = rj * 8 + fj
                mask[i, j] = True
    return mask


def build_king_mask() -> torch.Tensor:
    """(64, 64) bool — True if king-adjacent (8-connected, + self-connection)."""
    deltas = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]
    mask = torch.zeros(64, 64, dtype=torch.bool)
    for i in range(64):
        ri, fi = _sq_to_rf(i)
        mask[i, i] = True  # self-connection
        for dr, df in deltas:
            rj, fj = ri + dr, fi + df
            if 0 <= rj < 8 and 0 <= fj < 8:
                j = rj * 8 + fj
                mask[i, j] = True
    return mask


def build_all_static_masks() -> torch.Tensor:
    """
    Build all 6 static topology masks.

    Returns:
        (6, 64, 64) bool tensor.
        Index: 0=file, 1=rank, 2=diagonal, 3=anti_diagonal, 4=knight, 5=king
    """
    masks = torch.stack([
        build_file_mask(),
        build_rank_mask(),
        build_diagonal_mask(),
        build_anti_diagonal_mask(),
        build_knight_mask(),
        build_king_mask(),
    ], dim=0)  # (6, 64, 64)
    return masks


def build_square_delta_index_tables() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute signed square-offset indices for all query/key pairs.

    The tables are indexed as [query_square, key_square] and encode:
      delta_rank = key_rank - query_rank in [0, 14] after +7 shift
      delta_file = key_file - query_file in [0, 14] after +7 shift
    """
    rank_idx = torch.arange(8, dtype=torch.long).repeat_interleave(8)  # (64,)
    file_idx = torch.arange(8, dtype=torch.long).repeat(8)             # (64,)
    delta_rank = rank_idx.unsqueeze(0) - rank_idx.unsqueeze(1) + 7
    delta_file = file_idx.unsqueeze(0) - file_idx.unsqueeze(1) + 7
    return delta_rank, delta_file


_SQUARE_DELTA_RANK_IDX, _SQUARE_DELTA_FILE_IDX = build_square_delta_index_tables()


# =============================================================================
# Precomputed "between" tables for ray computation
# =============================================================================

def _build_between_table() -> Dict[Tuple[int, int], List[int]]:
    """
    For each pair of squares (i, j) that lie on the same file, rank, or diagonal,
    return the list of square indices strictly between them.

    Only includes pairs where i != j and they are aligned.
    """
    table = {}
    for i in range(64):
        ri, fi = _sq_to_rf(i)
        for j in range(i + 1, 64):
            rj, fj = _sq_to_rf(j)

            dr = rj - ri
            df = fj - fi

            # Check alignment: same file, same rank, or same diagonal
            aligned = False
            if df == 0 and dr != 0:
                aligned = True  # same file
            elif dr == 0 and df != 0:
                aligned = True  # same rank
            elif abs(dr) == abs(df) and dr != 0:
                aligned = True  # diagonal or anti-diagonal

            if not aligned:
                continue

            # Direction step
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_f = 0 if df == 0 else (1 if df > 0 else -1)

            between = []
            cr, cf = ri + step_r, fi + step_f
            while (cr, cf) != (rj, fj):
                between.append(cr * 8 + cf)
                cr += step_r
                cf += step_f

            table[(i, j)] = between
            table[(j, i)] = between  # symmetric

    return table


def _build_between_tensors() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert between table to tensor form for efficient batched computation.

    Returns:
        aligned_pairs: (N_pairs, 2) long — indices of aligned square pairs (i < j)
        between_indices: (N_total_between,) long — flat list of between-square indices
        pair_offsets: (N_pairs + 1,) long — offsets into between_indices for each pair
    """
    table = _build_between_table()
    # Deduplicate: only store (i, j) with i < j
    pairs = []
    between_flat = []
    offsets = [0]

    seen = set()
    for (i, j), between in table.items():
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
        between_flat.extend(between)
        offsets.append(len(between_flat))

    aligned_pairs = torch.tensor(pairs, dtype=torch.long)       # (N_pairs, 2)
    between_indices = torch.tensor(between_flat, dtype=torch.long)  # (N_total,)
    pair_offsets = torch.tensor(offsets, dtype=torch.long)       # (N_pairs + 1,)

    return aligned_pairs, between_indices, pair_offsets


# =============================================================================
# Dynamic mask builders
# =============================================================================

class DynamicMaskBuilder(nn.Module):
    """
    Computes dynamic (position-dependent) attention masks from board tensors.

    Fully vectorized — no Python-level loops over batch or squares.

    - Ray mask: unobstructed sliding path between aligned squares
    - Attack mask: piece on square i attacks square j
    """

    def __init__(self):
        super().__init__()
        # ── Ray mask precomputation ──────────────────────────────────
        aligned_pairs, between_indices, pair_offsets = _build_between_tensors()
        self.register_buffer('aligned_pairs', aligned_pairs)       # (N_pairs, 2)

        # Separate adjacent pairs (no between squares) from non-adjacent
        lengths = pair_offsets[1:] - pair_offsets[:-1]  # (N_pairs,)
        adj_mask = (lengths == 0)
        nonadj_mask = ~adj_mask

        # Adjacent pairs: always-clear, store as (N_adj, 2)
        self.register_buffer('adj_pairs', aligned_pairs[adj_mask])

        # Non-adjacent pairs: pad between-indices to max length
        nonadj_pairs = aligned_pairs[nonadj_mask]  # (N_nonadj, 2)
        self.register_buffer('nonadj_pairs', nonadj_pairs)

        nonadj_offsets = pair_offsets[:-1][nonadj_mask]
        nonadj_lengths = lengths[nonadj_mask]
        max_between = int(nonadj_lengths.max().item()) if nonadj_lengths.numel() > 0 else 0
        self.max_between = max_between

        if max_between > 0 and nonadj_pairs.size(0) > 0:
            n_nonadj = nonadj_pairs.size(0)
            # Padded between indices: (N_nonadj, max_between), pad with 0 (masked out)
            padded_between = torch.zeros(n_nonadj, max_between, dtype=torch.long)
            valid_mask = torch.zeros(n_nonadj, max_between, dtype=torch.bool)
            for i in range(n_nonadj):
                start = pair_offsets[:-1][nonadj_mask][i].item()
                ln = nonadj_lengths[i].item()
                padded_between[i, :ln] = between_indices[start:start + ln]
                valid_mask[i, :ln] = True
            self.register_buffer('padded_between', padded_between)   # (N_nonadj, max_between)
            self.register_buffer('between_valid', valid_mask)        # (N_nonadj, max_between)
        else:
            self.register_buffer('padded_between', torch.zeros(0, 1, dtype=torch.long))
            self.register_buffer('between_valid', torch.zeros(0, 1, dtype=torch.bool))

        # ── Attack mask precomputation ───────────────────────────────
        # Cache static alignment masks for sliding piece attack computation
        diag_mask = build_diagonal_mask() | build_anti_diagonal_mask()
        file_rank_mask = build_file_mask() | build_rank_mask()
        self.register_buffer('diag_mask', diag_mask)          # (64, 64)
        self.register_buffer('file_rank_mask', file_rank_mask) # (64, 64)

        # Precompute static attack patterns for non-sliding pieces: (64, 64) bool each
        self.register_buffer('wp_attacks', self._build_pawn_attacks(direction=+1))  # white pawn
        self.register_buffer('bp_attacks', self._build_pawn_attacks(direction=-1))  # black pawn
        self.register_buffer('knight_attacks', build_knight_mask())                  # includes self
        self.register_buffer('king_attacks', build_king_mask())                      # includes self

    @staticmethod
    def _build_pawn_attacks(direction: int) -> torch.Tensor:
        """Precompute pawn attack pattern. direction=+1 for white, -1 for black."""
        mask = torch.zeros(64, 64, dtype=torch.bool)
        for sq in range(64):
            r, f = sq // 8, sq % 8
            mask[sq, sq] = True  # self-connection
            tr = r + direction
            if 0 <= tr < 8:
                for df in [-1, 1]:
                    tf = f + df
                    if 0 <= tf < 8:
                        mask[sq, tr * 8 + tf] = True
        return mask

    def compute_ray_mask(self, boards: torch.Tensor) -> torch.Tensor:
        """
        Compute unobstructed sliding-path mask (vectorized).

        Args:
            boards: (B, 18, 8, 8)

        Returns:
            (B, 64, 64) bool — True if unobstructed sliding path between i and j
        """
        B = boards.size(0)
        device = boards.device

        occupancy = (boards[:, :12, :, :].sum(dim=1) > 0.5).reshape(B, 64)  # (B, 64)

        # Start with self-connections
        ray_mask = torch.eye(64, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        # Adjacent aligned pairs — always clear (no blockers possible)
        if self.adj_pairs.size(0) > 0:
            ai = self.adj_pairs[:, 0]  # (N_adj,)
            aj = self.adj_pairs[:, 1]
            ray_mask[:, ai, aj] = True
            ray_mask[:, aj, ai] = True

        # Non-adjacent pairs — check if all between squares are empty
        n_nonadj = self.nonadj_pairs.size(0)
        if n_nonadj > 0 and self.max_between > 0:
            # Gather occupancy at between squares: (B, N_nonadj, max_between)
            between_occ = occupancy[:, self.padded_between]  # (B, N_nonadj, max_between)
            # Mask out padding positions (treat as unoccupied)
            between_occ = between_occ & self.between_valid.unsqueeze(0)  # (B, N_nonadj, max_between)
            # Path is clear if no between-square is occupied
            clear = ~between_occ.any(dim=2)  # (B, N_nonadj)

            ni = self.nonadj_pairs[:, 0]  # (N_nonadj,)
            nj = self.nonadj_pairs[:, 1]
            # Scatter: ray_mask[:, ni, nj] = clear (batched advanced indexing)
            ray_mask[:, ni, nj] = clear
            ray_mask[:, nj, ni] = clear

        return ray_mask

    def compute_attack_mask(self, boards: torch.Tensor,
                             ray_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attack mask (vectorized): piece on square i attacks square j.

        Piece types (from board channels):
            0=wP, 1=wN, 2=wB, 3=wR, 4=wQ, 5=wK,
            6=bP, 7=bN, 8=bB, 9=bR, 10=bQ, 11=bK

        Args:
            boards: (B, 18, 8, 8)
            ray_mask: (B, 64, 64) bool, optional — pre-computed ray mask to avoid
                      redundant recomputation. If None, computed internally.

        Returns:
            (B, 64, 64) bool — True if piece on i attacks j
        """
        B = boards.size(0)
        device = boards.device

        piece_types = extract_piece_types(boards)  # (B, 64) values 0-12

        # Start with self-connections
        attack_mask = torch.eye(64, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1).clone()

        # Compute ray mask for sliding pieces (reuse if already computed)
        if ray_mask is None:
            ray_mask = self.compute_ray_mask(boards)  # (B, 64, 64)

        # --- Non-sliding pieces: broadcast precomputed static patterns ---
        # For each piece type, is_type is (B, 64) bool.
        # pattern is (64, 64) bool.
        # Contribution: is_type[:, :, None] & pattern[None, :, :] → (B, 64, 64)

        # White pawns (type 0)
        is_wp = (piece_types == 0).unsqueeze(2)   # (B, 64, 1)
        attack_mask = attack_mask | (is_wp & self.wp_attacks.unsqueeze(0))

        # Black pawns (type 6)
        is_bp = (piece_types == 6).unsqueeze(2)
        attack_mask = attack_mask | (is_bp & self.bp_attacks.unsqueeze(0))

        # Knights (type 1 or 7)
        is_knight = ((piece_types == 1) | (piece_types == 7)).unsqueeze(2)
        attack_mask = attack_mask | (is_knight & self.knight_attacks.unsqueeze(0))

        # Kings (type 5 or 11)
        is_king = ((piece_types == 5) | (piece_types == 11)).unsqueeze(2)
        attack_mask = attack_mask | (is_king & self.king_attacks.unsqueeze(0))

        # --- Sliding pieces: ray_mask intersected with alignment masks ---
        # Bishops (type 2 or 8): diagonal rays
        is_bishop = ((piece_types == 2) | (piece_types == 8)).unsqueeze(2)  # (B, 64, 1)
        bishop_rays = ray_mask & self.diag_mask.unsqueeze(0)                # (B, 64, 64)
        attack_mask = attack_mask | (is_bishop & bishop_rays)

        # Rooks (type 3 or 9): file/rank rays
        is_rook = ((piece_types == 3) | (piece_types == 9)).unsqueeze(2)
        rook_rays = ray_mask & self.file_rank_mask.unsqueeze(0)
        attack_mask = attack_mask | (is_rook & rook_rays)

        # Queens (type 4 or 10): all rays
        is_queen = ((piece_types == 4) | (piece_types == 10)).unsqueeze(2)
        attack_mask = attack_mask | (is_queen & ray_mask)

        return attack_mask


# =============================================================================
# Chess Structure Attention (structured + optional global heads)
# =============================================================================

class ChessStructureAttention(nn.Module):
    """
    Multi-head attention with per-head chess topology masks.

    Heads 0-5: static relations (file, rank, diagonal, anti-diagonal, knight, king)
    Head 6: dynamic ray mask (unobstructed sliding path)
    Head 7: dynamic attack mask (piece attacks)
    Heads 8+: global attention (all squares visible)

    All heads include self-connections.

    Relative-position modes:
      - none:
          masks define which edges may communicate; attention scores are otherwise
          content-only.
      - score_bias:
          adds a learned relative score bias indexed by (head, delta_rank, delta_file).
          This changes routing only and keeps value vectors unchanged.
      - edge_modulation:
          builds a learned edge embedding from static geometry plus head identity and
          uses it to modulate both attention scores and value messages.
          This is more expressive, but it uses a custom attention path instead of SDPA.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_ray_mask: bool = True,
        use_attack_mask: bool = True,
        relative_mode: str = "none",
        relative_edge_dim: int = 16,
        ablation_no_mask: bool = False,
    ):
        super().__init__()
        assert n_heads >= 8, f"ChessStructureAttention requires at least 8 heads, got {n_heads}"
        if relative_mode not in {"none", "score_bias", "edge_modulation"}:
            raise ValueError(
                "relative_mode must be one of {'none', 'score_bias', 'edge_modulation'} "
                f"(got {relative_mode!r})"
            )
        if int(relative_edge_dim) <= 0:
            raise ValueError(f"relative_edge_dim must be > 0 (got {relative_edge_dim})")
        self.ablation_no_mask = ablation_no_mask
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0, f"dim {dim} not divisible by n_heads {n_heads}"
        self.use_ray_mask = use_ray_mask
        self.use_attack_mask = use_attack_mask
        self.relative_mode = relative_mode
        self.relative_edge_dim = int(relative_edge_dim)
        self.scale = math.sqrt(self.head_dim)

        # Q/K/V/O projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Register static masks: (6, 64, 64) bool
        static_masks = build_all_static_masks()
        self.register_buffer('static_masks', static_masks)
        self.register_buffer('delta_rank_idx', _SQUARE_DELTA_RANK_IDX.clone())
        self.register_buffer('delta_file_idx', _SQUARE_DELTA_FILE_IDX.clone())
        self.register_buffer('head_indices', torch.arange(n_heads, dtype=torch.long))

        self.relative_score_bias: Optional[nn.Parameter] = None
        self.edge_rank_embedding: Optional[nn.Embedding] = None
        self.edge_file_embedding: Optional[nn.Embedding] = None
        self.edge_head_embedding: Optional[nn.Embedding] = None
        self.edge_proj: Optional[nn.Linear] = None
        self.edge_to_k: Optional[nn.Linear] = None
        self.edge_to_v: Optional[nn.Linear] = None

        if self.relative_mode == "score_bias":
            # Zero init keeps score_bias identical to the baseline until trained.
            self.relative_score_bias = nn.Parameter(torch.zeros(n_heads, 15, 15))
        elif self.relative_mode == "edge_modulation":
            # v1 uses only static geometry plus head identity in the edge embedding.
            # Dynamic ray/attack information remains a hard mask in this first pass.
            edge_dim = self.relative_edge_dim
            self.edge_rank_embedding = nn.Embedding(15, edge_dim)
            self.edge_file_embedding = nn.Embedding(15, edge_dim)
            self.edge_head_embedding = nn.Embedding(n_heads, edge_dim)
            self.edge_proj = nn.Linear(edge_dim * 3, edge_dim)
            self.edge_to_k = nn.Linear(edge_dim, self.head_dim)
            self.edge_to_v = nn.Linear(edge_dim, self.head_dim * 2)
            nn.init.zeros_(self.edge_to_k.weight)
            nn.init.zeros_(self.edge_to_k.bias)
            nn.init.zeros_(self.edge_to_v.weight)
            nn.init.zeros_(self.edge_to_v.bias)

    def forward(
        self,
        x: torch.Tensor,
        dynamic_ray_mask: Optional[torch.Tensor] = None,
        dynamic_attack_mask: Optional[torch.Tensor] = None,
        head_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 64, dim)
            dynamic_ray_mask: (B, 64, 64) bool, optional (ignored if head_masks provided)
            dynamic_attack_mask: (B, 64, 64) bool, optional (ignored if head_masks provided)
            head_masks: (B, n_heads, 64, 64) bool, optional — pre-built combined mask.
                        When provided, skips per-layer mask rebuild.

        Returns:
            (B, 64, dim)
        """
        B, S, D = x.shape  # S=64
        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2)  # (B, H, 64, Dh)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2)

        # Build per-head mask if not pre-computed
        if head_masks is None:
            head_masks = self._build_head_masks(
                B, x.device, dynamic_ray_mask, dynamic_attack_mask
            )

        if self.relative_mode == "none":
            # Baseline path: masks are the only structural bias. Routing among allowed
            # edges is content-only and can stay on the fast SDPA bool-mask path.
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=head_masks,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )  # (B, H, 64, Dh)
        elif self.relative_mode == "score_bias":
            # score_bias changes routing only: the learned relative term is added to
            # the attention logits, but the value vectors remain untouched.
            attn_bias = self._build_relative_score_bias(head_masks, q.dtype)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )  # (B, H, 64, Dh)
        else:
            # edge_modulation changes both routing and message semantics. The edge
            # embedding modulates K for pair-specific scores and applies a FiLM-style
            # transform to V, so this path uses a custom attention computation.
            attn_out = self._forward_edge_modulation(q, k, v, head_masks)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(attn_out)

    def _build_relative_score_bias(
        self,
        head_masks: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.relative_score_bias is None:
            raise RuntimeError("score_bias mode requires relative_score_bias to be initialized")
        bias = self.relative_score_bias[:, self.delta_rank_idx, self.delta_file_idx]  # (H, 64, 64)
        bias = bias.unsqueeze(0).expand(head_masks.size(0), -1, -1, -1).to(dtype=dtype)
        return bias.masked_fill(~head_masks, float('-inf'))

    def _build_edge_embeddings(self, dtype: torch.dtype) -> torch.Tensor:
        if (
            self.edge_rank_embedding is None
            or self.edge_file_embedding is None
            or self.edge_head_embedding is None
            or self.edge_proj is None
        ):
            raise RuntimeError("edge_modulation mode requires edge embeddings to be initialized")

        rank_embed = self.edge_rank_embedding(self.delta_rank_idx)        # (64, 64, E)
        file_embed = self.edge_file_embedding(self.delta_file_idx)        # (64, 64, E)
        head_embed = self.edge_head_embedding(self.head_indices)          # (H, E)
        head_embed = head_embed[:, None, None, :].expand(-1, 64, 64, -1)
        rank_embed = rank_embed.unsqueeze(0).expand(self.n_heads, -1, -1, -1)
        file_embed = file_embed.unsqueeze(0).expand(self.n_heads, -1, -1, -1)
        edge_input = torch.cat([rank_embed, file_embed, head_embed], dim=-1)
        return self.edge_proj(edge_input).to(dtype=dtype)                # (H, 64, 64, E)

    def _forward_edge_modulation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        head_masks: torch.Tensor,
    ) -> torch.Tensor:
        if self.edge_to_k is None or self.edge_to_v is None:
            raise RuntimeError("edge_modulation mode requires edge_to_k and edge_to_v to be initialized")

        edge_repr = self._build_edge_embeddings(q.dtype)                 # (H, 64, 64, E)
        edge_k = self.edge_to_k(edge_repr)                               # (H, 64, 64, Dh)
        edge_v = self.edge_to_v(edge_repr)                               # (H, 64, 64, 2*Dh)
        gamma, beta = edge_v.chunk(2, dim=-1)                            # (H, 64, 64, Dh)

        base_scores = torch.matmul(q, k.transpose(-2, -1)).float() / float(self.scale)
        edge_scores = torch.einsum("bhid,hijd->bhij", q, edge_k).float() / float(self.scale)
        scores = (base_scores + edge_scores).masked_fill(~head_masks, float('-inf'))

        weights = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        if self.training and self.attn_dropout.p > 0.0:
            weights = F.dropout(weights, p=self.attn_dropout.p)

        value_scale = 1.0 + 0.1 * torch.tanh(gamma)
        scaled_values = torch.einsum("bhij,bhjd,hijd->bhid", weights, v, value_scale)
        shifted_values = torch.einsum("bhij,hijd->bhid", weights, beta)
        return scaled_values + shifted_values

    def _build_head_masks(
        self,
        batch_size: int,
        device: torch.device,
        dynamic_ray_mask: Optional[torch.Tensor],
        dynamic_attack_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build (B, n_heads, 64, 64) bool mask for all heads.

        Heads 0-5: static masks (broadcast over batch)
        Head 6: dynamic ray mask (or global if not provided)
        Head 7: dynamic attack mask (or global if not provided)
        Heads 8+: global attention

        If ablation_no_mask is True, returns all-True (fully-connected) masks.
        """
        if self.ablation_no_mask:
            return torch.ones(
                batch_size, self.n_heads, 64, 64,
                dtype=torch.bool, device=device,
            )

        # Static heads: (6, 64, 64) → (1, 6, 64, 64) → broadcast to (B, 6, 64, 64)
        static = self.static_masks.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Dynamic heads
        # Head 6: ray mask
        if self.use_ray_mask and dynamic_ray_mask is not None:
            ray = dynamic_ray_mask.unsqueeze(1)  # (B, 1, 64, 64)
        else:
            # Global attention (all-True) as fallback
            ray = torch.ones(batch_size, 1, 64, 64, dtype=torch.bool, device=device)

        # Head 7: attack mask
        if self.use_attack_mask and dynamic_attack_mask is not None:
            attack = dynamic_attack_mask.unsqueeze(1)  # (B, 1, 64, 64)
        else:
            attack = torch.ones(batch_size, 1, 64, 64, dtype=torch.bool, device=device)

        masks = torch.cat([static, ray, attack], dim=1)  # (B, 8, 64, 64)

        # Extra heads (>8) are global attention
        if self.n_heads > 8:
            global_heads = torch.ones(
                batch_size,
                self.n_heads - 8,
                64,
                64,
                dtype=torch.bool,
                device=device,
            )
            masks = torch.cat([masks, global_heads], dim=1)

        return masks  # (B, n_heads, 64, 64)


# =============================================================================
# Chess Structure Layer (Pre-LN residual block)
# =============================================================================

class ChessStructureLayer(nn.Module):
    """One layer of chess-topology sparse attention + FFN with Pre-LN."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        use_ray_mask: bool = True,
        use_attack_mask: bool = True,
        relative_mode: str = "none",
        relative_edge_dim: int = 16,
        ablation_no_mask: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ChessStructureAttention(
            dim, n_heads, dropout, use_ray_mask, use_attack_mask,
            relative_mode=relative_mode,
            relative_edge_dim=relative_edge_dim,
            ablation_no_mask=ablation_no_mask,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        dynamic_ray_mask: Optional[torch.Tensor] = None,
        dynamic_attack_mask: Optional[torch.Tensor] = None,
        head_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 64, dim)
            dynamic_ray_mask: (B, 64, 64) bool, optional (ignored if head_masks provided)
            dynamic_attack_mask: (B, 64, 64) bool, optional (ignored if head_masks provided)
            head_masks: (B, n_heads, 64, 64) bool, optional — pre-built combined mask

        Returns:
            (B, 64, dim)
        """
        x = x + self.attn(self.norm1(x), head_masks=head_masks)
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# Chess Structure Message Passing (full module)
# =============================================================================

class ChessStructureMP(nn.Module):
    """
    Full Chess Structure Message Passing module.

    Input composition per square:
        taper_mlp(cat(raw_cnn_features))[cnn_proj_dim] || pos_embed[pos_dim] || piece_embed[piece_dim]
        -> project to csmp_dim
        -> N layers of ChessStructureLayer
        -> project to output_dim (tap_dim)

    Args:
        cnn_dim: Per-tap CNN feature dim (typically 256)
        num_taps: Number of CNN taps
        output_dim: Output dimension (tap_dim for Perceiver input)
        csmp_dim: Internal working dimension
        pos_dim: Positional embedding dimension
        piece_dim: Piece-type embedding dimension
        cnn_proj_dim: If set, concatenate all taps then compress via a
                      tapering MLP (halving dims each layer) from
                      num_taps*cnn_dim down to cnn_proj_dim. Allows
                      cross-tap interaction during compression.
                      If None, raw concatenated CNN dims are used.
        n_layers: Number of CSMP layers
        n_heads: Number of attention heads (must be >= 8)
        ffn_mult: FFN expansion multiplier
        dropout: Dropout rate
        use_ray_mask: Enable dynamic ray masking
        use_attack_mask: Enable dynamic attack masking
        use_xy_coords: Concatenate normalized (rank, file) coordinates to positional
                       embeddings and project back to pos_dim.
        relative_mode: Mutually exclusive CSMP relative-position mode.
                       none = masks only, score_bias = additive relative logits,
                       edge_modulation = pair-conditioned key/value modulation.
        relative_edge_dim: Edge embedding dim used only by relative_mode=edge_modulation.
        ablation_no_mask: If True, disable all per-head chess masks and use
                          fully-connected multihead attention (ablation mode).
    """

    def __init__(
        self,
        cnn_dim: int = 256,
        num_taps: int = 4,
        output_dim: int = 1024,
        csmp_dim: int = 1024,
        pos_dim: int = 32,
        piece_dim: int = 64,
        cnn_proj_dim: Optional[int] = None,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        use_ray_mask: bool = True,
        use_attack_mask: bool = True,
        use_xy_coords: bool = False,
        relative_mode: str = "none",
        relative_edge_dim: int = 16,
        ablation_no_mask: bool = False,
    ):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.num_taps = num_taps
        self.csmp_dim = csmp_dim
        self.pos_dim = pos_dim
        self.piece_dim = piece_dim
        self.cnn_proj_dim = cnn_proj_dim
        self.use_ray_mask = use_ray_mask
        self.use_attack_mask = use_attack_mask
        self.use_xy_coords = use_xy_coords
        self.relative_mode = relative_mode
        self.relative_edge_dim = int(relative_edge_dim)
        self.ablation_no_mask = ablation_no_mask

        # --- CNN taper MLP (optional dimensionality reduction) ---
        # Concatenates all taps then gradually compresses via halving MLP
        raw_cnn_dim = num_taps * cnn_dim  # e.g. 4*256 = 1024, or 0 if no CNN
        if raw_cnn_dim > 0 and cnn_proj_dim is not None and cnn_proj_dim > 0 and cnn_proj_dim < raw_cnn_dim:
            layers = []
            d_in = raw_cnn_dim
            # Halve until we'd overshoot the target
            while d_in // 2 >= cnn_proj_dim:
                d_out = max(d_in // 2, cnn_proj_dim)
                layers.extend([
                    nn.Linear(d_in, d_out),
                    nn.LayerNorm(d_out),
                    nn.GELU(),
                ])
                d_in = d_out
                if d_in == cnn_proj_dim:
                    break
            self.cnn_taper = nn.Sequential(*layers)
            effective_cnn_dim = cnn_proj_dim
        else:
            self.cnn_taper = None
            effective_cnn_dim = raw_cnn_dim

        # --- Learned embeddings ---
        # Positional: rank(8) -> pos_dim/2, file(8) -> pos_dim/2
        half_pos = pos_dim // 2
        self.rank_embedding = nn.Embedding(8, half_pos)
        self.file_embedding = nn.Embedding(8, half_pos)
        if self.use_xy_coords:
            self.pos_coord_proj = nn.Linear(pos_dim + 2, pos_dim)
        else:
            self.pos_coord_proj = None

        # Piece-type: 13 types (12 pieces + empty)
        self.piece_embedding = nn.Embedding(13, piece_dim)

        # --- Input projection ---
        input_dim = effective_cnn_dim + pos_dim + piece_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, csmp_dim),
            nn.LayerNorm(csmp_dim),
        )

        # --- CSMP layers ---
        self.layers = nn.ModuleList([
            ChessStructureLayer(
                csmp_dim, n_heads, ffn_mult, dropout,
                use_ray_mask, use_attack_mask,
                relative_mode=relative_mode,
                relative_edge_dim=self.relative_edge_dim,
                ablation_no_mask=ablation_no_mask,
            )
            for _ in range(n_layers)
        ])

        # --- Output projection ---
        self.output_proj = nn.Sequential(
            nn.Linear(csmp_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # --- Dynamic mask builder ---
        self.dynamic_masks = DynamicMaskBuilder()

        # Print summary
        total_params = sum(p.numel() for p in self.parameters())
        if self.cnn_taper is not None:
            taper_dims = [raw_cnn_dim]
            for m in self.cnn_taper:
                if isinstance(m, nn.Linear):
                    taper_dims.append(m.out_features)
            taper_str = '->' .join(str(d) for d in taper_dims)
            cnn_desc = f"taper({taper_str})"
        else:
            cnn_desc = f"{num_taps}x{cnn_dim}={raw_cnn_dim}"
        print(f"  [CSMP] {n_layers} layers, {n_heads} heads, dim={csmp_dim}")
        print(f"    Input: {cnn_desc} CNN + {pos_dim} pos + {piece_dim} piece "
              f"= {input_dim} -> {csmp_dim}")
        print(f"    Positional coords: normalized_xy={use_xy_coords}")
        print(
            f"    Relative position mode: {self.relative_mode}"
            + (
                f" (edge_dim={self.relative_edge_dim})"
                if self.relative_mode == "edge_modulation"
                else ""
            )
        )
        print(f"    Output: {csmp_dim} -> {output_dim}")
        print(f"    Dynamic masks: ray={use_ray_mask}, attack={use_attack_mask}")
        if ablation_no_mask:
            print(f"    *** ABLATION: all chess-specific masks disabled (fully-connected attention) ***")
        print(f"    Total params: {total_params:,}")

    def _build_pos_embed(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Build per-square positional embeddings.

        Returns: (B, 64, pos_dim)
        """
        rank_idx = torch.arange(8, device=device).repeat_interleave(8)  # (64,)
        file_idx = torch.arange(8, device=device).repeat(8)              # (64,)

        rank_emb = self.rank_embedding(rank_idx)  # (64, pos_dim/2)
        file_emb = self.file_embedding(file_idx)  # (64, pos_dim/2)

        pos = torch.cat([rank_emb, file_emb], dim=-1)  # (64, pos_dim)
        if self.use_xy_coords:
            rank_xy = rank_idx.to(pos.dtype).unsqueeze(1) / 7.0
            file_xy = file_idx.to(pos.dtype).unsqueeze(1) / 7.0
            coords = torch.cat([rank_xy, file_xy], dim=-1)  # (64, 2), normalized [0, 1]
            pos = torch.cat([pos, coords], dim=-1)  # (64, pos_dim + 2)
            pos = self.pos_coord_proj(pos)  # (64, pos_dim)
        return pos.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 64, pos_dim)

    def _build_piece_embed(
        self, boards: torch.Tensor
    ) -> torch.Tensor:
        """
        Build per-square piece-type embeddings from board tensor.

        Args:
            boards: (B, 18, 8, 8)

        Returns: (B, 64, piece_dim)
        """
        piece_types = extract_piece_types(boards)  # (B, 64)
        return self.piece_embedding(piece_types)     # (B, 64, piece_dim)

    def forward(
        self,
        cnn_tap_features: List[torch.Tensor],
        boards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cnn_tap_features: List of (B, 256, 8, 8) tensors, one per tapped CNN block.
                              Will be reshaped to (B, 64, 256) per-square tokens.
            boards: (B, 18, 8, 8) — original board tensor for piece-type extraction
                    and dynamic mask computation.

        Returns:
            (B, 64, output_dim) — enriched per-square features for the Perceiver
        """
        B = boards.size(0)
        device = boards.device

        # CSMP profiling: CUDA event timing (activated by adapter when profiling.csmp_timing=True)
        _profile_csmp = getattr(self, '_profile', False)
        if _profile_csmp:
            _ev_csmp_s = torch.cuda.Event(enable_timing=True)
            _ev_csmp_e = torch.cuda.Event(enable_timing=True)
            _ev_csmp_s.record()

        # --- Build embeddings ---
        pos_embed = self._build_pos_embed(B, device)  # (B, 64, pos_dim)
        piece_embed = self._build_piece_embed(boards)   # (B, 64, piece_dim)

        # --- Reshape CNN features (if any) and compose input ---
        if len(cnn_tap_features) > 0:
            tap_spatials = []
            for cnn_out in cnn_tap_features:
                # (B, 256, 8, 8) -> (B, 8, 8, 256) -> (B, 64, 256)
                spatial = cnn_out.permute(0, 2, 3, 1).reshape(B, 64, self.cnn_dim)
                tap_spatials.append(spatial)

            # Concatenate all taps per square: (B, 64, N_taps * cnn_dim)
            cnn_concat = torch.cat(tap_spatials, dim=-1)

            # Compress via taper MLP if configured
            if self.cnn_taper is not None:
                cnn_concat = self.cnn_taper(cnn_concat)  # (B, 64, cnn_proj_dim)

            ref_dtype = cnn_concat.dtype
            pos_embed = pos_embed.to(ref_dtype)
            piece_embed = piece_embed.to(ref_dtype)
            x = torch.cat([cnn_concat, pos_embed, piece_embed], dim=-1)
        else:
            # CNN-free mode: input is only positional + piece embeddings
            ref_dtype = pos_embed.dtype
            piece_embed = piece_embed.to(ref_dtype)
            x = torch.cat([pos_embed, piece_embed], dim=-1)

        x = self.input_proj(x)  # (B, 64, csmp_dim)

        # --- Compute dynamic masks once (reuse across all layers) ---
        dynamic_ray_mask = None
        dynamic_attack_mask = None

        if self.use_ray_mask or self.use_attack_mask:
            dynamic_ray_mask = self.dynamic_masks.compute_ray_mask(boards)
        if self.use_attack_mask:
            # Pass pre-computed ray_mask to avoid redundant recomputation
            dynamic_attack_mask = self.dynamic_masks.compute_attack_mask(
                boards, ray_mask=dynamic_ray_mask
            )

        # Build combined head masks once for all layers: (B, n_heads, 64, 64) bool
        head_masks = self.layers[0].attn._build_head_masks(
            B, device,
            dynamic_ray_mask if self.use_ray_mask else None,
            dynamic_attack_mask,
        )

        # --- CSMP layers ---
        for layer in self.layers:
            x = layer(x, head_masks=head_masks)

        # --- Output projection ---
        x = self.output_proj(x)  # (B, 64, output_dim)

        if _profile_csmp:
            _ev_csmp_e.record()
            torch.cuda.synchronize(device)
            self._last_ms = _ev_csmp_s.elapsed_time(_ev_csmp_e)

        return x
