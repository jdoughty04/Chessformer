# Structured Square Mixer Math

This document gives the exact forward equations and routing regularizers for
`xattn_mode: structured_square_mixer` in the chess-fusion adapter.

It covers:

- the decoder-side routing equations
- the $64 \times 3$ aligned slot construction
- the square-marginal sparsity loss
- the square-diversity loss
- the precise config / live-control scalars that weight these terms

## 1. Notation

For one decoder fusion layer, let:

- $B$ = batch size
- $T$ = number of text tokens in the current decoder layer input
- $d_l$ = LLM hidden size
- $d_r$ = recurrent-query state size
- $d_p$ = Perceiver latent size
- $i \in \{0, \dots, 63\}$ = board-square index
- $s \in \{\text{csmp}, \text{perc}, \text{pol}\}$ = square-aligned source index
- $k \in \{\text{perc\_global}, \text{side}\}$ = global-branch source index

The layer receives:

- $h_{b,t} \in \mathbb{R}^{d_l}$: decoder hidden state for batch item $b$, token $t$
- $x_i^{\text{csmp}} \in \mathbb{R}^{d_{\text{ctx}}}$: CSMP square token for square $i$
- $x_i^{\text{perc}} \in \mathbb{R}^{d_p}$: Perceiver square latent for square $i$
- $x_i^{\text{pol}} \in \mathbb{R}^{d_p}$: structured policy latent for square $i$
- $g_b^{\text{perc}} \in \mathbb{R}^{d_p}$: Perceiver global latent
- $g_b^{\text{side}} \in \mathbb{R}^{d_{\text{ctx}}}$: side-to-move token from pre-Perceiver context

The valid-text-token mask is:

- $m_{b,t} \in \{0,1\}$

All token averages below are taken only over valid positions:

- $|V| = \sum_{b,t} m_{b,t}$

## 2. Text-Side Query State

The decoder token is first converted into a recurrent query state:

$$
r_{b,1:T} = \text{GRU}(\text{LN}(h_{b,1:T}))
$$

where:

- $r_{b,t} \in \mathbb{R}^{d_r}$
- invalid positions are zeroed after the GRU:
  $r_{b,t} \leftarrow 0$ when $m_{b,t} = 0$

## 3. Slot-Weight Logits

Each text token scores the $64 \times 3 = 192$ aligned square slots using the
concatenation of:

- its text-side recurrent state $r_{b,t}$
- the Perceiver global latent $g_b^{\text{perc}}$

Define:

$$
q_{b,t} = [r_{b,t} ; g_b^{\text{perc}}] \in \mathbb{R}^{d_r + d_p}
$$

Then the slot logits are:

$$
z_{b,t} = W_{\text{sq}} q_{b,t} + b_{\text{sq}} \in \mathbb{R}^{192}
$$

The normalized slot weights are:

$$
\alpha_{b,t} = \text{softmax}(z_{b,t})
$$

We index them as:

$$
\alpha_{b,t,s,i}
$$

with the constraint:

$$
\sum_s \sum_i \alpha_{b,t,s,i} = 1
$$

for every valid token $(b,t)$.

## 4. Value Construction For The $64 \times 3$ Aligned Slots

Each square source is projected into LLM space with a source-specific MLP:

- $v_{b,i}^{\text{csmp}} = \text{MLP}_{\text{csmp}}(x_{b,i}^{\text{csmp}}) \in \mathbb{R}^{d_l}$
- $v_{b,i}^{\text{perc}} = \text{MLP}_{\text{perc}}(x_{b,i}^{\text{perc}}) \in \mathbb{R}^{d_l}$
- $v_{b,i}^{\text{pol}}  = \text{MLP}_{\text{pol}}(x_{b,i}^{\text{pol}}) \in \mathbb{R}^{d_l}$

These are concatenated into one aligned slot table:

$$
V_b = [v_{b,0}^{\text{csmp}}, \dots, v_{b,63}^{\text{csmp}},
       v_{b,0}^{\text{perc}}, \dots, v_{b,63}^{\text{perc}},
       v_{b,0}^{\text{pol}},  \dots, v_{b,63}^{\text{pol}}] \in \mathbb{R}^{192 \times d_l}
$$

The square-mixer output for token $(b,t)$ is:

$$
m_{b,t}^{\text{sq}} = \sum_s \sum_i \alpha_{b,t,s,i} v_{b,i}^s \in \mathbb{R}^{d_l}
$$

## 5. Global Branch

The structured mixer also keeps a separate 2-way global branch. Its values are:

- $u_b^{\text{perc\_global}} = \text{MLP}_{\text{perc\_global}}(g_b^{\text{perc}}) \in \mathbb{R}^{d_l}$
- $u_b^{\text{side}} = \text{MLP}_{\text{side}}(g_b^{\text{side}}) \in \mathbb{R}^{d_l}$

Its logits come from the recurrent text state alone:

$$
z_{b,t}^{\text{glb}} = W_{\text{glb}} r_{b,t} + b_{\text{glb}} \in \mathbb{R}^2
$$

$$
\beta_{b,t} = \text{softmax}(z_{b,t}^{\text{glb}})
$$

with components:

- $\beta_{b,t,\text{perc\_global}}$
- $\beta_{b,t,\text{side}}$

The global-branch mix is:

$$
m_{b,t}^{\text{glb}} = \beta_{b,t,\text{perc\_global}} u_b^{\text{perc\_global}}
              + \beta_{b,t,\text{side}} u_b^{\text{side}}
$$

## 6. Decoder Update

The two branches are added:

$$
m_{b,t} = m_{b,t}^{\text{sq}} + m_{b,t}^{\text{glb}}
$$

This mixed vector is then reshaped into heads, gated, projected, and added as a
residual update:

$$
o_{b,t} = O( \tanh(\text{gate}) \cdot \text{reshape\_heads}(m_{b,t}) )
$$

$$
h'_{b,t} = h_{b,t} + o_{b,t}
$$

Then the layer applies its FFN residual:

$$
h''_{b,t} = h'_{b,t} + \tanh(\text{ffn\_gate}) \cdot \text{FFN}(\text{LN}(h'_{b,t}))
$$

## 7. Why The Routing Regularizer Is Square-Marginal

The router distribution $\alpha_{b,t,s,i}$ is over $192$ slots, but the desired
inductive bias is:

- sparse over board squares
- not necessarily diverse over the three source types

So the regularizer first marginalizes out source identity:

$$
p_{b,t,i}^{\text{sq}} = \sum_s \alpha_{b,t,s,i}
$$

Now:

$$
\sum_i p_{b,t,i}^{\text{sq}} = 1
$$

and $p_{b,t}^{\text{sq}}$ is a pure 64-square distribution.

This is the key design choice: routing can still prefer one source over another
for a given square, but the auxiliary losses only care about which squares are
being used.

## 8. Square-Marginal Sparsity Loss

For each valid token, compute the 64-square entropy:

$$
H_{b,t}^{\text{sq}} = - \sum_i p_{b,t,i}^{\text{sq}} \log p_{b,t,i}^{\text{sq}}
$$

Average over valid text tokens:

$$
H_{\text{sq}} = \frac{1}{|V|} \sum_{b,t} m_{b,t} H_{b,t}^{\text{sq}}
$$

Normalize by the maximum entropy $\log 64$:

$$
L_{\text{sparse}} = \frac{H_{\text{sq}}}{\log 64}
$$

Properties:

- $L_{\text{sparse}}$ is near $0$ when each token puts its mass on very few squares
- $L_{\text{sparse}}$ is near $1$ when each token spreads mass nearly uniformly over all
  64 squares

Config/live-control scalar:

- `structured_xattn_sparse_weight`

Contribution to total auxiliary loss:

$$
\lambda_{\text{sparse}} L_{\text{sparse}}
$$

where:

- $\lambda_{\text{sparse}}$ = `structured_xattn_sparse_weight`

## 9. Square-Usage Diversity Loss

Token-level sparsity alone can collapse the whole layer onto one square. To
counter that, we also track the mean square usage across valid text tokens:

$$
u_i = \frac{1}{|V|} \sum_{b,t} m_{b,t} p_{b,t,i}^{\text{sq}}
$$

So:

$$
\sum_i u_i = 1
$$

and $u \in \mathbb{R}^{64}$ is the aggregate square-usage distribution for the layer.

Its entropy is:

$$
H_{\text{usage}} = - \sum_i u_i \log u_i
$$

Normalized:

$$
H_{\text{usage\_norm}} = \frac{H_{\text{usage}}}{\log 64}
$$

The diversity penalty is a floor, not a matching-to-uniform loss:

$$
L_{\text{div}} = \max(0, \tau - H_{\text{usage\_norm}})
$$

where:

- $\tau \in [0,1]$ is the target normalized entropy floor

Interpretation:

- if square usage is already diverse enough ($H_{\text{usage\_norm}} \ge \tau$), then
  $L_{\text{div}} = 0$
- if usage collapses too far, the loss grows linearly as entropy falls below the
  target

Config/live-control scalars:

- `structured_xattn_square_diversity_weight`
- `structured_xattn_square_diversity_target_entropy`

Contribution to total auxiliary loss:

$$
\lambda_{\text{div}} L_{\text{div}}
$$

where:

- $\lambda_{\text{div}}$ = `structured_xattn_square_diversity_weight`
- $\tau$ = `structured_xattn_square_diversity_target_entropy`

## 10. Total Routing Auxiliary Loss

The structured-mixer routing contribution is:

$$
L_{\text{router}} = \lambda_{\text{sparse}} L_{\text{sparse}} + \lambda_{\text{div}} L_{\text{div}}
$$

This is added into the model's larger auxiliary objective together with policy,
move-eval, BSR, and SPP losses:

$$
L_{\text{aux\_total}} = \dots + \lambda_{\text{sparse}} L_{\text{sparse}} + \lambda_{\text{div}} L_{\text{div}}
$$

Important:

- the sparse term acts on per-token square marginals
- the diversity term acts on the layer-level mean square usage
- neither term directly enforces balance across the three source types

## 11. Logged Metrics

The code exposes both the old slot-level diagnostics and the new square-level
diagnostics.

### Slot-level diagnostics

These still describe the full $192$-way routing distribution:

- `slot_mean`
- `slot_entropy`
- `source_mass`
- `max_slot_index`
- `max_slot_source_index`
- `max_slot_square_index`
- `max_slot_mass`

### Square-level diagnostics

These describe the source-marginalized 64-square routing:

- `square_mean_i` $= \frac{1}{|V|} \sum_{b,t} m_{b,t} p_{b,t,i}^{\text{sq}}$
- `square_entropy`
- `square_entropy_norm`
- `square_usage_entropy`
- `square_usage_entropy_norm`
- `max_square_index`
- `max_square_mass`

The most directly useful live metrics are:

- `structured_xattn/.../square_entropy_norm`
  - low means token-level square routing is sharp
- `structured_xattn/.../square_usage_entropy_norm`
  - low means the layer is collapsing onto a small set of squares
- `train/structured_xattn_sparse_loss`
  - the normalized per-token square entropy actually used in loss
- `train/structured_xattn_square_diversity_loss`
  - the hinge penalty below the target diversity floor
- `train/structured_xattn_square_usage_entropy`
  - the current normalized aggregate square-usage entropy

## 12. Practical Reading Of The Two Regularizers

The intended regime is:

- $L_{\text{sparse}}$ low enough that each token chooses a small number of squares
- $H_{\text{usage\_norm}}$ high enough that the whole layer does not always choose the
  same square

In other words:

- sparsity answers "how many squares does one token use?"
- diversity answers "do different tokens spread their attention over different
  squares?"

That combination is why the current design uses:

- square-marginal sparsity
- square-usage diversity
- no explicit source-diversity regularizer

The source types are semantically asymmetric (`csmp`, `perc`, `pol`), so
forcing them toward equal usage would be a different inductive bias from the one
implemented here.
