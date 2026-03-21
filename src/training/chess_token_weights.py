"""Chess token loss weighting.

Classifies tokenized text into chess-semantic categories (squares, pieces,
moves, sides, files) and builds a per-token weight tensor.  Tokens that carry
factual board-state information receive higher loss weight so the LM objective
penalises hallucinated squares / pieces more than filler words.

The weight tensor has the same shape as ``input_ids`` and is 1.0 for ordinary
tokens.  Chess-critical tokens get the configured multiplier instead.

Usage (inside dataset ``__getitem__``)::

    weights = build_chess_token_loss_weights(input_ids, tokenizer, cfg)
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from training.config import TrainingConfig


# ── Precompiled patterns ────────────────────────────────────────────────

# Chess squares: a1-h8
_SQUARE_RE = re.compile(r"^[a-h][1-8]$")

# Piece names (lowercase, full words only)
_PIECE_NAMES = frozenset({
    "king", "queen", "rook", "bishop", "knight", "pawn",
    "kings", "queens", "rooks", "bishops", "knights", "pawns",
})

# Side references
_SIDE_TOKENS = frozenset({
    "white", "black",
    "white's", "black's",
})

# File references: "a-file" through "h-file" (the full hyphenated token or
# the letter part when the tokenizer splits on the hyphen).
_FILE_RE = re.compile(r"^[a-h]-file$")
# Standalone file letters when followed by "-file" are handled via bigram
# lookahead in the builder (see below).

# SAN move patterns ────────────────────────────────────────────────────
# Covers: e4, exd5, Nf3, Bxe5, Qd1, Kc2, Rfe1, N3d4, Qh4+, Bxf7#,
#         O-O, O-O-O, e8=Q, promotion variants
_SAN_MOVE_RE = re.compile(
    r"^(?:"
    r"O-O-O|O-O"                              # castling
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8][+#]?" # piece moves
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?"    # pawn captures (exd5, exd8=Q)
    r"|[a-h][1-8](?:=[QRBN])?[+#]?"           # pawn pushes (e4, e8=Q)
    r")$"
)

# We also want to catch sub-tokens that are clearly part of a SAN move but
# got split by BPE.  For example "Bx" or "Nf" or "exd" or "O-O" fragments.
# These are handled by the _is_san_fragment helper.
_SAN_PIECE_PREFIX_RE = re.compile(r"^[KQRBN][a-h]?[1-8]?x?$")
_SAN_PAWN_CAPTURE_PREFIX_RE = re.compile(r"^[a-h]x[a-h]?$")


def _is_san_fragment(tok_str: str) -> bool:
    """Return True if *tok_str* looks like a BPE fragment of a SAN move."""
    if _SAN_PIECE_PREFIX_RE.match(tok_str):
        return True
    if _SAN_PAWN_CAPTURE_PREFIX_RE.match(tok_str):
        return True
    # Promotion suffixes
    if tok_str in ("=Q", "=R", "=B", "=N"):
        return True
    # Check/mate symbols attached to a square
    if tok_str in ("+", "#"):
        return True
    return False


# ── Token decoding cache ────────────────────────────────────────────────

@lru_cache(maxsize=8192)
def _decode_token(tokenizer_id: int, tok_id: int, tokenizer) -> str:
    """Decode a single token id to a stripped string.

    Results are cached per (tokenizer identity, token_id) to avoid repeated
    decode calls across samples.
    """
    return tokenizer.decode([tok_id]).strip()


def _get_tokenizer_id(tokenizer) -> int:
    """Stable identity for the tokenizer (used as cache key)."""
    return id(tokenizer)


# ── Public API ──────────────────────────────────────────────────────────

def build_chess_token_loss_weights(
    input_ids: torch.Tensor,
    tokenizer: "PreTrainedTokenizerBase",
    config: "TrainingConfig",
) -> torch.Tensor:
    """Build a per-token loss weight tensor for *input_ids*.

    Parameters
    ----------
    input_ids : (seq_len,) int tensor
        Token IDs for a single sample (1-D, no batch dim).
    tokenizer : HuggingFace tokenizer
        Used to decode individual tokens for classification.
    config : TrainingConfig
        Provides the weight multipliers and the ``chess_token_weight_enabled`` flag.

    Returns
    -------
    loss_weights : (seq_len,) float32 tensor
        1.0 for ordinary tokens, higher for chess-critical tokens.
    """
    seq_len = input_ids.shape[0]
    weights = torch.ones(seq_len, dtype=torch.float32)

    if not config.chess_token_weight_enabled:
        return weights

    w_square = config.chess_token_weight_squares
    w_piece = config.chess_token_weight_pieces
    w_move = config.chess_token_weight_moves
    w_side = config.chess_token_weight_sides
    w_file = config.chess_token_weight_files

    tok_id_key = _get_tokenizer_id(tokenizer)

    # Decode all tokens once
    decoded = []
    for i in range(seq_len):
        tid = int(input_ids[i].item())
        s = _decode_token(tok_id_key, tid, tokenizer)
        decoded.append(s)

    for i, tok_str in enumerate(decoded):
        if not tok_str:
            continue

        tok_lower = tok_str.lower()

        # 1. Full SAN moves (highest priority — checked first)
        #    These often contain squares too (e.g. "Nf3" contains f3),
        #    but we want the move weight to dominate.
        if _SAN_MOVE_RE.match(tok_str):
            weights[i] = w_move
            continue

        # 2. Squares: a1-h8
        if _SQUARE_RE.match(tok_lower):
            weights[i] = w_square
            continue

        # 3. Piece names
        if tok_lower in _PIECE_NAMES:
            weights[i] = w_piece
            continue

        # 4. Side references
        if tok_lower in _SIDE_TOKENS:
            weights[i] = w_side
            continue

        # 5. File references: "a-file", "h-file", etc.
        if _FILE_RE.match(tok_lower):
            weights[i] = w_file
            continue
        # Handle split file tokens: if this token is a single file letter
        # and the *next* token starts with "-file", weight both.
        if len(tok_lower) == 1 and tok_lower in "abcdefgh":
            if i + 1 < seq_len:
                next_tok = decoded[i + 1].lower()
                if next_tok.startswith("-file"):
                    weights[i] = w_file
                    weights[i + 1] = w_file
                    continue

        # 6. SAN move fragments (BPE splits like "Bx", "Nf", "exd")
        if _is_san_fragment(tok_str):
            weights[i] = w_move
            continue

    return weights
