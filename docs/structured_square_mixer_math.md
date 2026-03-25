# Structured Cross-Attn Math

This document describes the exact forward equations and structured
square-attention regularizers for the chess-fusion adapter's canonical
structured decoder fusion mode:

- `xattn_mode: structured_cross_attn`

It covers:

- hidden-state query formation
- the $64 \times N_{\mathrm{src}}$ aligned square-slot construction
- per-head square and global cross-attention
- token-conditioned gating
- square sparsity, diversity, and gate-usage regularizers

## 1. Notation

For one decoder fusion layer, let:

- $B$: batch size
- $T$: number of text tokens at that decoder layer
- $d_\ell$: LLM hidden size
- $H$: number of x-attn heads
- $d_h = d_\ell / H$: per-head dimension
- $i \in \{0, \ldots, 63\}$: board-square index
- $s \in S$: square-aligned source index
- $k \in \{\mathrm{percGlobal}, \mathrm{side}\}$: global-branch source index

By default:

- $S = \{\mathrm{csmp}, \mathrm{perc}, \mathrm{pol}\}$
- $N_{\mathrm{src}} = 3$

If `xattn_structured_use_engineered_source: true`, then:

- $S = \{\mathrm{csmp}, \mathrm{perc}, \mathrm{pol}, \mathrm{eng}\}$
- $N_{\mathrm{src}} = 4$

If `engineered_only_xattn_ablation: true`, then:

- $S = \{\mathrm{eng}\}$
- $N_{\mathrm{src}} = 1$
- the learned backbone / CSMP / Perceiver sources are skipped
- the global latent and side token become deterministic summaries derived from
  engineered features plus side-to-move

The layer receives:

- $h_{b,t} \in \mathbb{R}^{d_\ell}$: decoder hidden state for token $t$
- $x_{b,i}^{\mathrm{csmp}}$: CSMP square token for square $i$
- $x_{b,i}^{\mathrm{perc}}$: Perceiver square latent for square $i$
- $x_{b,i}^{\mathrm{pol}}$: structured policy latent for square $i$
- $x_{b,i}^{\mathrm{eng}}$: optional engineered square feature vector
- $g_b^{\mathrm{perc}}$: Perceiver global latent
- $g_b^{\mathrm{side}}$: side-to-move / context token
- $m_{b,t} \in \{0,1\}$: valid-text-token mask

The `pol` source is the same 64-square readout shared with the move-level auxiliary heads. Those heads score Maia's move vocabulary by applying head-specific `from` / `to` projections to this square table and pairing the corresponding endpoints.

All token averages below are taken only over valid text positions:

- $|V| = \sum_{b,t} m_{b,t}$

## 2. Hidden-State Queries

There is no recurrent text path. Each token query comes directly from the
current decoder hidden state:

$$
\tilde{q}_{b,t} = \mathrm{LN}(h_{b,t})
$$

The fused multi-head query tensor is:

$$
Q_{b,t,h} = \mathrm{reshape}_{H}(W_q \tilde{q}_{b,t}) \in \mathbb{R}^{d_h}
$$

Invalid text positions are masked out after query formation, so they contribute
zero attention output and zero gate usage.

## 3. Structured Square Values

Each active square source is projected independently into LLM space:

$$
v_{b,i}^{\mathrm{csmp}} = \mathrm{MLP}_{\mathrm{csmp}}(x_{b,i}^{\mathrm{csmp}}) \in \mathbb{R}^{d_\ell}
$$

$$
v_{b,i}^{\mathrm{perc}} = \mathrm{MLP}_{\mathrm{perc}}(x_{b,i}^{\mathrm{perc}}) \in \mathbb{R}^{d_\ell}
$$

$$
v_{b,i}^{\mathrm{pol}} = \mathrm{MLP}_{\mathrm{pol}}(x_{b,i}^{\mathrm{pol}}) \in \mathbb{R}^{d_\ell}
$$

When enabled:

$$
v_{b,i}^{\mathrm{eng}} = \mathrm{Linear}_{\mathrm{eng}}(\mathrm{LN}(x_{b,i}^{\mathrm{eng}})) \in \mathbb{R}^{d_\ell}
$$

These are concatenated into one aligned square table:

$$
V_b^{\mathrm{sq}} =
\left[
v_{b,0}^{s_{1}}, \ldots, v_{b,63}^{s_{1}},
v_{b,0}^{s_{2}}, \ldots,
v_{b,63}^{s_{N_{\mathrm{src}}}}
\right]
$$

so $V_b^{\mathrm{sq}} \in \mathbb{R}^{(64 N_{\mathrm{src}}) \times d_\ell}$.

The same projected values are also mapped to attention keys:

$$
K_b^{\mathrm{sq}} = \mathrm{reshape}_{H}(W_k^{\mathrm{sq}} V_b^{\mathrm{sq}})
\in \mathbb{R}^{(64 N_{\mathrm{src}}) \times H \times d_h}
$$

$V_b^{\mathrm{sq}}$ itself is reshaped per head as:

$$
V_{b,j,h}^{\mathrm{sq}} \in \mathbb{R}^{d_h},
\qquad
j \in \{0, \ldots, 64 N_{\mathrm{src}} - 1\}
$$

## 4. Global Values

The structured path keeps a separate 2-token global branch:

$$
u_b^{\mathrm{percGlobal}} = \mathrm{MLP}_{\mathrm{percGlobal}}(g_b^{\mathrm{perc}}) \in \mathbb{R}^{d_\ell}
$$

$$
u_b^{\mathrm{side}} = \mathrm{MLP}_{\mathrm{side}}(g_b^{\mathrm{side}}) \in \mathbb{R}^{d_\ell}
$$

Stack them as:

$$
V_b^{\mathrm{glb}} = \left[u_b^{\mathrm{percGlobal}}, u_b^{\mathrm{side}}\right]
\in \mathbb{R}^{2 \times d_\ell}
$$

and project keys:

$$
K_b^{\mathrm{glb}} = \mathrm{reshape}_{H}(W_k^{\mathrm{glb}} V_b^{\mathrm{glb}})
\in \mathbb{R}^{2 \times H \times d_h}
$$

## 5. Square Attention Branch

Each head attends over the full $64 \times N_{\mathrm{src}}$ aligned square table:

$$
a_{b,t,h,j}^{\mathrm{sq}} =
\frac{\langle Q_{b,t,h}, K_{b,j,h}^{\mathrm{sq}} \rangle}{\sqrt{d_h}}
$$

$$
\alpha_{b,t,h,j}^{\mathrm{sq}} = \mathrm{softmax}_{j}(a_{b,t,h,j}^{\mathrm{sq}})
$$

where the softmax runs over all square/source slots $j$.

The per-head square mixture is:

$$
M_{b,t,h}^{\mathrm{sq}} =
\sum_j \alpha_{b,t,h,j}^{\mathrm{sq}} V_{b,j,h}^{\mathrm{sq}}
$$

If we unpack slot $j$ into $(s, i)$, then:

$$
\alpha_{b,t,h,s,i}^{\mathrm{sq}},
\qquad
\sum_s \sum_i \alpha_{b,t,h,s,i}^{\mathrm{sq}} = 1
$$

## 6. Global Attention Branch

Each head also attends over the two global tokens:

$$
a_{b,t,h,k}^{\mathrm{glb}} =
\frac{\langle Q_{b,t,h}, K_{b,k,h}^{\mathrm{glb}} \rangle}{\sqrt{d_h}}
$$

$$
\beta_{b,t,h,k}^{\mathrm{glb}} = \mathrm{softmax}_{k}(a_{b,t,h,k}^{\mathrm{glb}})
$$

$$
M_{b,t,h}^{\mathrm{glb}} =
\sum_k \beta_{b,t,h,k}^{\mathrm{glb}} V_{b,k,h}^{\mathrm{glb}}
$$

The square and global branches are kept separate up to the gated sum. The
regularizers described below apply only to the square branch, never to the
2-token global branch.

## 7. Token-Conditioned Gates And Residual Injection

Each head has a learned static gate logit $g_h^{\mathrm{static}}$.

If `xattn_text_gate_mode: tanh_head`, the layer also predicts token-conditioned
gate logits directly from the normalized decoder hidden states:

$$
\ell_{b,t,h} = \mathrm{MLP}_{\mathrm{gate}}(\tilde{q}_{b,t})
$$

The effective structured gate is:

- without token gates:

$$
g_{b,t,h}^{\mathrm{eff}} = m_{b,t} \cdot \tanh(g_h^{\mathrm{static}})
$$

- with token gates:

$$
g_{b,t,h}^{\mathrm{eff}} = m_{b,t} \cdot \tanh(g_h^{\mathrm{static}} + \ell_{b,t,h})
$$

The combined per-head structured output is:

$$
O_{b,t,h} =
g_{b,t,h}^{\mathrm{eff}} \cdot \left(M_{b,t,h}^{\mathrm{sq}} + M_{b,t,h}^{\mathrm{glb}}\right)
$$

Heads are concatenated, projected back to model width, and added residually:

$$
O_{b,t} = W_o \mathrm{concat}_{h}(O_{b,t,h})
$$

$$
h'_{b,t} = h_{b,t} + O_{b,t}
$$

The fusion layer then applies its gated FFN residual in the usual adapter path.

## 8. Square Marginals

The raw square/source attention is over $64 N_{\mathrm{src}}$ slots, but the square
regularizers operate on the source-marginalized 64-square distribution:

$$
p_{b,t,h,i} = \sum_s \alpha_{b,t,h,s,i}^{\mathrm{sq}}
$$

This is the key design choice: attention may prefer one source over another for
the same square, while sparsity/diversity act only on "which squares matter?"

## 9. Sparse Square-Attention Metric

For each valid token/head pair, define the normalized square entropy:

$$
H_{\mathrm{norm}}(p_{b,t,h}) =
-\frac{\sum_i p_{b,t,h,i} \log p_{b,t,h,i}}{\log 64}
$$

The cached sparse metric for one layer is:

$$
L_{\mathrm{sparse}}^{\mathrm{layer}} =
\frac{1}{|V| H}
\sum_{b,t,h}
\left|g_{b,t,h}^{\mathrm{eff}}\right|
H_{\mathrm{norm}}(p_{b,t,h})
$$

This is exactly what `structured_xattn_sparse_weight` multiplies in the total
loss. Lower values mean sharper square attention on tokens/heads where the
structured gate is actually open.

## 10. Square-Diversity Metric

First compute a gate-weighted average square-usage distribution for each head:

$$
w_{b,t,h} = \left|g_{b,t,h}^{\mathrm{eff}}\right|
$$

$$
\bar{p}_{h,i} =
\frac{\sum_{b,t} w_{b,t,h} p_{b,t,h,i}}
{\sum_{b,t} w_{b,t,h}}
$$

Then compute the normalized usage entropy per head:

$$
U_h = -\frac{\sum_i \bar{p}_{h,i} \log \bar{p}_{h,i}}{\log 64}
$$

The layer-level usage entropy cached for logging is the gate-weighted average:

$$
U_{\mathrm{layer}} = \frac{\sum_h W_h U_h}{\sum_h W_h}
$$

where

$$
W_h = \sum_{b,t} w_{b,t,h}
$$

The diversity loss for one layer is:

$$
L_{\mathrm{div}}^{\mathrm{layer}} =
\mathrm{mean}_{b,t,h}(w_{b,t,h})
\cdot
\mathrm{relu}(\tau_{\mathrm{entropy}} - U_{\mathrm{layer}})
$$

with $\tau_{\mathrm{entropy}} =$ `structured_xattn_square_diversity_target_entropy`,
clamped to $[0, 1]$.

This keeps the regularizer weak when the structured path is mostly closed, while
penalizing collapse to the same few squares once the model is actively using the
structured branch.

## 11. Gate-Usage Loss

The mean absolute gate usage for one layer is:

$$
G_{\mathrm{layer}} = \mathrm{mean}_{b,t,h}\left|g_{b,t,h}^{\mathrm{eff}}\right|
$$

The corresponding hinge loss is:

$$
L_{\mathrm{gate}}^{\mathrm{layer}} =
\mathrm{relu}(\tau_{\mathrm{usage}} - G_{\mathrm{layer}})
$$

with $\tau_{\mathrm{usage}} =$ `structured_xattn_gate_usage_target`, clamped to $[0, 1]$.

## 12. Loss Composition Across Layers

Across all active fusion layers in `structured_cross_attn` mode:

- `structured_xattn_sparse_loss` is the mean of $L_{\mathrm{sparse}}^{\mathrm{layer}}$
- `structured_xattn_square_diversity_loss` is the mean of $L_{\mathrm{div}}^{\mathrm{layer}}$
- `structured_xattn_gate_usage_loss` is the mean of $L_{\mathrm{gate}}^{\mathrm{layer}}$

The training loss adds, where $\lambda_{\mathrm{sparse}}$,
$\lambda_{\mathrm{div}}$, and $\lambda_{\mathrm{gate}}$ are the configured
weights `structured_xattn_sparse_weight`,
`structured_xattn_square_diversity_weight`, and
`structured_xattn_gate_usage_weight`:

$$
\lambda_{\mathrm{sparse}} \, \mathrm{structuredXattnSparseLoss}
\;+\;
\lambda_{\mathrm{div}} \, \mathrm{structuredXattnSquareDiversityLoss}
\;+\;
\lambda_{\mathrm{gate}} \, \mathrm{structuredXattnGateUsageLoss}
$$

## 13. Inspector And Logging Tensors

The structured inspector and training logs are derived from real attention
weights, not router logits.

Important cached quantities include:

- `slot_mean`: mean attention over all $64 N_{\mathrm{src}}$ square slots
- `source_mass`: source-marginal mass for each active square source
- `square_mean`: 64-square source-marginal attention distribution
- `global_mean`: mean 2-way global attention distribution
- `effective_gate_abs_mean`: mean absolute effective gate usage
- `token_gate_logit_mean`: mean token-conditioned gate logit when enabled

The decode inspector additionally caches last-token traces for:

- mean square attention
- per-head square attention
- gate-weighted attention views
- normalized post-gate contribution norms after `o_proj`

So the GUI answers "which squares and sources did this token read from?" using
the actual structured cross-attention executed at runtime.
