# Training Guide

## Running Training

```bash
python src/training/train.py --config configs/chess_fusion.yaml
```

The config file controls everything — model architecture, training hyperparameters, which components to freeze, and which objectives to enable.

## Training Regimes

The same config file supports multiple training regimes by toggling a few key flags:

### Full Commentary Training (LLM + Adapter)

The default mode. The chess adapter encodes the position, auxiliary chess heads provide grounding losses, and the LLM generates commentary.

```yaml
model:
  enable_lm: true
  use_lora: true
```

### Adapter-Only Pretraining (No LLM)

Train just the chess encoder and auxiliary heads. Useful for building chess representations before connecting to the LLM.

```yaml
model:
  enable_lm: false
```

The training loop automatically switches to `PolicyOnlyModel` when `enable_lm: false`.

### Progressive Unfreezing

Components can be frozen/unfrozen independently. A typical progression:

1. **Phase 1**: Train adapter only (freeze everything else)
2. **Phase 2**: Unfreeze LoRA and cross-attention
3. **Phase 3**: Optionally unfreeze CSMP/Perceiver for fine-tuning

Freeze flags in the config:
```yaml
chess_fusion:
  freeze_csmp: true
  freeze_perceiver: true
  freeze_xattn: false
  freeze_prepend_latents: false
```

LoRA can also start frozen and unfreeze at a specified epoch:
```yaml
chess_fusion:
  lora:
    start_frozen: true
    unfreeze_epoch: 2
```

A live control server (default port 8585) allows freeze/unfreeze toggling during training without restarting.

## Config Walkthrough

The config is organized into three sections:

### `training:` — Loop and Data

| Key | What it does |
|-----|-------------|
| `samples_dir` | Path to exported `.pt` training samples |
| `resume_from_checkpoint` | Path to resume from a previous run |
| `batch_size`, `gradient_accumulation_steps` | Effective batch = batch_size × accumulation |
| `learning_rate`, `warmup_ratio` | Base LR with linear warmup |
| `chess_token_weight_*` | Upweight loss on chess-specific tokens (squares, pieces, moves) |
| `secondary_samples_dir`, `secondary_mix_ratio` | Optional second dataset for data mixing |

### `model:` — Architecture

| Key | What it does |
|-----|-------------|
| `base_model` | HuggingFace model ID (default: TinyLlama 1.1B) |
| `enable_lm` | `true` for full commentary, `false` for adapter-only |
| `use_lora` | Enable LoRA adaptation on the LLM |
| `chess_fusion.*` | All encoder/fusion architecture settings (see config comments) |

### `profiling:` — Performance Monitoring

Optional CUDA timing, torch profiler traces, memory snapshots, and MFU calculation. All no-ops when `enabled: false`.

## Checkpointing

- Checkpoints save every `save_steps` steps
- Selective module loading via `load_checkpoint_*` flags lets you resume specific components
- WandB logging tracks all losses, routing diagnostics, and validation generations

## Objectives and Loss Weights

All objective weights are configurable. Set weight to `0.0` to disable any head.

| Objective | Config key | Purpose |
|-----------|-----------|---------|
| LM loss | (always on when `enable_lm: true`) | Next-token commentary prediction |
| Policy distillation | `aux_policy_weight` | Match Maia move probabilities |
| Move-eval ranking | `aux_move_eval_weight` | Rank candidate moves by engine score |
| Move-eval regression | `move_eval_mse_weight` | Predict centipawn values |
| BSR | `bsr_weight` | Reconstruct piece identity per square |
| SPP | `spp_weight` | Predict attack counts and ray features |
| Router sparsity | `structured_xattn_sparse_weight` | Keep decoder routing focused on few squares |
| Router diversity | `structured_xattn_square_diversity_weight` | Prevent routing collapse to same squares |
| Router gate usage | `structured_xattn_gate_usage_weight` | Keep token-conditioned chess injection from collapsing fully off |

For structured fusion runs, the most important routing controls are usually:

- `xattn_structured_router_mode`: `shared` keeps one square router per token; `per_head` lets each x-attn head pick its own squares.
- `xattn_text_gate_mode`: `tanh_head` adds a token-conditioned per-head gate on top of the learned static head gates.
- `structured_xattn_gate_usage_target`: sets the minimum average `|effective_gate|` encouraged by the weak hinge loss.
