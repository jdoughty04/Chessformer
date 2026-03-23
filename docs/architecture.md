# Architecture

While the best performing neural chess engines traditionally use full 64x64 attention mechanisms (and often CNN stems), specializing the architecture is a promising approach. This paper [Enhancing Chess Reinforcement Learning with Graph Representations](https://arxiv.org/pdf/2410.23753) found that sparse, relation-specific attention mechanisms have higher learning affinity in lower-compute training regimes.

This approach uses expressive self-attention mechanisms to learn positional properties, but structurally enforces biases toward square-level representations so that squares encode information and relationships that are directly related to them. The idea is that keeping square-level representations:
- Naturally adheres to a useful inductive bias for chess, potentially improving optimization and transferability
- Allows for better mechanistic interpretability and debugging as we can probe how the model processes information at the square level
-

## Pipeline

1. Represent the board as 64 square tokens plus side-to-move context
2. Inject explicit chess topology via structure-aware message passing
3. Compress into structured latents (one per square + one global)
4. Train those latents with chess-native objectives
5. Fuse chess state into selected LLM decoder layers

<p align="center">
  <img src="../assets/model_architecture.svg" alt="ChessFormer model architecture diagram" width="980">
</p>

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

CSMP supports three modes for optional relative position encoding schemes that influence how square pairs interact:

| Mode | Behavior |
|------|----------|
| `none` | Masks-only baseline; routing is purely content-driven |
| `score_bias` | Adds learned bias indexed by `(head, delta_rank, delta_file)`; changes routing, not values |
| `edge_modulation` | Learned edge embeddings modulate both keys and values; more expressive but more expensive |

Source: `src/training/chess_structure_mp.py`

## 3. Structured Perceiver Latents

`SquareLatentEncoder` is a Perceiver-style bottleneck with **65 structured latents**: 64 square-aligned plus 1 global.

Each Perceiver block runs latent self-attention, cross-attention to the chess context, and an FFN update. With `strict_own_square` masking, each square latent only cross-attends to its corresponding square token from the CSMP, along with the side token. Self attention remains global, and the global latent's cross attention is fully connected. The own-square masking preserves square identity while allowing some global interaction.

A subsequent readout, `_structured_policy_square_readout()`, produces 64 policy latents; also using strict-own-square masking, this is shared between several auxiliary prediction heads.

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

The active structured fusion mode is `structured_cross_attn`. Each selected decoder layer forms text queries from
`LayerNorm(hidden_states) -> q_proj`, then runs two per-head attention branches:

- square attention over $64 \times 3$ aligned slots (CSMP, Perceiver, Policy per square)
- global attention over the Perceiver global latent vs the side-to-move token
- optional token-conditioned per-head gate logits from the same normalized token states

Optionally, `xattn_structured_use_engineered_source: true` adds a fourth square-aligned source built from the `main` engineered features extracted in `src/training/chess_adapter.py`. Those `205` channels are:

- `64` dims: one-hot square identity
- `13` dims: piece occupancy by piece type/color plus an explicit empty-square channel
- `64` dims: attacked-target bitmask for the piece on that square
- `64` dims: defended-friendly-target bitmask for the piece on that square

In that mode the structured square table becomes $64 \times 4$, and the inspector exposes an additional `Engineered` board.

There is also a strong ablation mode: `engineered_only_xattn_ablation: true`.
In that configuration, the adapter backbone, CSMP stack, Perceiver, prepended
latents, pseudotokens, and auxiliary adapter heads are all skipped. Structured
x-attn is trained against a single square-aligned source:

- `Engineered`

The fusion layer still receives a global conditioning vector and a side/context
token, but in this ablation they are deterministic summaries derived directly
from the engineered features plus side-to-move rather than learned adapter
latents. That keeps the experiment focused on whether x-attn alone can learn to
route grounded square-local engineered features into the LLM.

With the structured source enabled, those same `205` channels stay directly
available at fusion time instead of being used only at the front of the model.

Runtime structured fusion uses per-head attention. The inspector can show
either individual heads or aggregate views by averaging or gate-weighting the
per-head attention maps.

The token-conditioned gate path (`xattn_text_gate_mode: tanh_head`) lets the model reduce chess injection on a token-by-token basis instead of relying only on the static learned head gates. In structured mode the effective injection gate is:

$$
\tanh(g_h^{\mathrm{static}} + \ell_{b,t,h})
$$

where $g_h^{\mathrm{static}}$ is the learned per-head static gate and
$\ell_{b,t,h}$ is the token-conditioned gate logit.

That dynamic gate is produced from the same normalized token state that also
forms the attention query, so "which chess features matter?" and "how much
chess should I inject?" stay coupled to the same decoder context.

The LLM reads the shared policy latents and generic Perceiver/CSMP sources, not the branch-specific prediction heads. This keeps one square-aligned representation central to both chess supervision and text generation.

The engineered source is intentionally simpler than the learned CSMP / Perceiver / Policy sources. That makes it useful as a grounding anchor and ablation, but it is still limited: the current features are mostly static occupancy and attack/defense structure. If grounded attention remains weak, the next feature improvements worth testing are attacker/defender counts by side or piece type, pin/check indicators, legal-move participation, ray-blocker structure, pawn-structure flags, and possibly a small separate global engineered token rather than forcing every global fact through square-local channels.

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
