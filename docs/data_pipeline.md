# Data Pipeline

## Overview

Training data flows through a two-repo pipeline:

```
PGN games
    |  (Synthetic Commentary Generation repo)
    v
Engine analysis + motif detection + importance scoring
    |
    v
LLM-synthesized commentary (Gemini)
    |
    v
Exported .pt samples
    |  (this repo)
    v
Training loop
```

The [Synthetic Commentary Generation](https://github.com/TODO/synthetic_commentary_generation) repository handles everything up to `.pt` export. This repo consumes those samples.

## Sample Contract

Each `.pt` training sample is a dictionary with:

**Required fields:**
- `fen` — board position in FEN notation
- `commentary` — the target text for LM training

**Optional fields (consumed when present):**
- `maia_policy` — precomputed Maia move probabilities (avoids running teacher during training)
- `move_evals` — per-move engine evaluations for ranking/regression heads
- `stockfish_eval_cp` — position evaluation in centipawns
- `stockfish_best_moves` — engine's top moves
- `pgn_moves` — game move history
- `last_move` — the move that led to this position
- `sample_id`, `task`, `source`, `metadata` — bookkeeping

The loader is intentionally permissive — the data-export side can evolve metadata without breaking training.

Source: `src/training/sample_contract.py`

## Data Mixing

The config supports optional secondary dataset mixing for grounding retention:

```yaml
training:
  samples_dir: "primary_samples"
  secondary_samples_dir: "grounding_samples"
  secondary_mix_ratio: 0.10
  secondary_mix_seed: 42
```
