# Architecture

The core idea is to keep square-level structure alive for as long as possible, rather than collapsing a chess position into a single vector before the LLM sees it.

## Pipeline

1. Represent the board as 64 square tokens plus side-to-move context
2. Inject explicit chess topology via structure-aware message passing
3. Compress into structured latents (one per square + one global)
4. Train those latents with chess-native objectives
5. Fuse chess state into selected LLM decoder layers

## 1. Input Representation

Every position is encoded as an 18-channel `8x8` board tensor (piece placement, side to move, castling rights, en passant). The board-only encoder builds square tokens directly from learned positional and piece embeddings; no pretrained CNN backbone is required.

Source: `src/training/maia_model.py`

## 2. Chess Structure Message Passing (CSMP)

`ChessStructureMP` runs sparse multi-head attention over the 64 square tokens. Each attention head is assigned to a chess-specific relation:

- Same file / rank / diagonal / anti-diagonal
- Knight connectivity
- King-neighborhood connectivity
- Dynamic sliding rays (based on occupancy)
- Dynamic attack relations (based on piece type and blockers)

This gives the model a board graph that encodes legal geometry before the Perceiver layers start compressing.

### Relative Position Modes

CSMP supports three modes for how square pairs interact:

| Mode | Behavior |
|------|----------|
| `none` | Masks-only baseline; routing is purely content-driven |
| `score_bias` | Adds learned bias indexed by `(head, delta_rank, delta_file)`; changes routing, not values |
| `edge_modulation` | Learned edge embeddings modulate both keys and values; more expressive but more expensive |

Source: `src/training/chess_structure_mp.py`

## 3. Structured Perceiver Latents

`SquareLatentEncoder` is a Perceiver-style bottleneck with **65 structured latents**: 64 square-aligned plus 1 global.

Each Perceiver block runs latent self-attention, cross-attention to the chess context, and an FFN update. With `strict_own_square` masking, each square latent primarily reads its own square token, while the global latent aggregates across the full board. This preserves square identity throughout.

A second readout, `_structured_policy_square_readout()`, produces 64 policy latents; the shared representation used by both the auxiliary chess heads and the LLM fusion bridge.

Source: `src/training/chess_fusion_model.py`

## 4. Auxiliary Chess Heads

The latent space is trained with several chess objectives. All prediction branches share the same 64 policy latents from the Perceiver readout, then specialize via their own `StructuredSquareBranchLayer`.

| Head | What it predicts |
|------|------------------|
| **Policy distillation** | Maia move probabilities (1880-move vocabulary, from/to dot product scoring) |
| **Move-eval ranking** | Soft CE + pairwise ranking over candidate moves |
| **Move-eval regression** | MSE on centipawn scores + mate classification |
| **BSR** (Board State Reconstruction) | 13-class per-square piece identity |
| **SPP** (Square Property Prediction) | 10 numeric channels per square (attack counts, ray distances) |

These losses let the adapter learn chess structure even when the LLM is frozen or disabled.

## 5. LLM Fusion

Selected decoder layers are wrapped with `FusionDecoderLayer`, which injects chess context via gated cross-attention.

The active fusion mode is **structured square mixer**. A shared router-conditioning MLP consumes the recurrent text state together with the Perceiver global latent, then emits:

- square-router logits over `64 x 3` aligned slots (CSMP, Perceiver, Policy per square)
- global-router logits over the Perceiver global latent vs the side-to-move token
- optional token-conditioned per-head gate logits

By default (`xattn_structured_router_mode: shared`), one `192`-way square router is shared across all x-attn heads for a token. That is the backward-compatible path and keeps the aggregate inspector view easy to read. With `xattn_structured_router_mode: per_head`, each x-attn head gets its own square/global router so different heads can specialize on different squares or source types.

The token-conditioned gate path (`xattn_text_gate_mode: tanh_head`) lets the model reduce chess injection on a token-by-token basis instead of relying only on the static learned head gates. In structured mode the effective injection gate is:

`tanh(static_head_gate + token_gate_logit)`

That dynamic gate is intentionally produced from the same router stem as the square/global routing decisions, so "which chess features matter?" and "how much chess should I inject?" are conditioned on the same token state.

The LLM reads the shared policy latents and generic Perceiver/CSMP sources, not the branch-specific prediction heads. This keeps one square-aligned representation central to both chess supervision and text generation.

Additional injection mechanisms:

- **Prepended latents**: chess latents projected into the LLM embedding space as prefix tokens
- **Layer pseudotokens**: per-layer learned KV pairs added to the attention context

For the exact forward equations and regularizer definitions, see [structured_square_mixer_math.md](structured_square_mixer_math.md).

Source: `src/training/chess_fusion_model.py`

## Supported LLM Backends

The fusion layer supports common HuggingFace decoder stacks:

- LLaMA-family (default: TinyLlama 1.1B with LoRA)
- GPT-NeoX-family
- GPT-2-style
