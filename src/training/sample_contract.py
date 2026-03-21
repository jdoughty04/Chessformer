"""Shared `.pt` sample contract helpers for training-time dataset loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def normalize_training_sample(
    sample: dict[str, Any],
    *,
    sample_path: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize a loaded `.pt` sample into the permissive training contract."""

    if not isinstance(sample, dict):
        raise TypeError(f"Training sample must be a dict, got {type(sample)!r}")

    normalized = dict(sample)
    fen = normalized.get("fen", "")
    commentary = normalized.get("commentary", "")
    pgn_moves = normalized.get("pgn_moves", "")

    if isinstance(commentary, dict):
        commentary = commentary.get("commentary_text", "")
    if not isinstance(commentary, str):
        commentary = str(commentary or "")
    if not isinstance(pgn_moves, str):
        pgn_moves = str(pgn_moves or "")
    if not isinstance(fen, str):
        fen = str(fen or "")

    if not fen.strip():
        where = f" in {sample_path}" if sample_path is not None else ""
        raise ValueError(f"Training sample is missing a non-empty `fen`{where}")
    if not commentary.strip():
        where = f" in {sample_path}" if sample_path is not None else ""
        raise ValueError(f"Training sample is missing a non-empty `commentary`{where}")

    normalized["fen"] = fen
    normalized["commentary"] = commentary
    normalized["pgn_moves"] = pgn_moves
    return normalized


def load_training_sample(path: str | Path) -> dict[str, Any]:
    """Load one `.pt` sample and normalize it to the training contract."""

    sample_path = Path(path)
    sample = torch.load(sample_path, weights_only=False)
    return normalize_training_sample(sample, sample_path=sample_path)
