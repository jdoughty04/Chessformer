# Structured Decode Inspector

The structured decode inspector is a local browser GUI for stepping through
commentary generation one token at a time while visualizing how a
`chess_fusion` checkpoint is routing into board-aligned chess state.

This tool is implemented in
`src/inference/decoding_inspector.py`.

## When To Use It

Use the inspector when you want to answer questions like:

- which token the model currently wants to emit next
- how peaked or uncertain the next-token distribution is
- which squares a selected cross-attention layer used to produce the last token
- whether a layer is reading more from CSMP, Perceiver, or Policy latents

This is primarily an interpretability and debugging tool for
`xattn_mode: structured_square_mixer`.

## Supported Checkpoints

The GUI currently supports only checkpoints that satisfy both conditions:

- `model.mode: chess_fusion`
- `model.chess_fusion.xattn_mode: structured_square_mixer`

It fails fast on other checkpoint types because the square heatmaps rely on the
structured `64 x 3` routing layout.

## Launching The GUI

From the `chess_fusion_training` directory:

```bash
python src/inference/decoding_inspector.py --checkpoint <checkpoint_dir>
```

Optional flags:

- `--port 8765`
- `--load-lora` or `--no-load-lora`
- `--use-merged-base`
- `--prior-checkpoints <ckpt1> <ckpt2> ...`

On startup the script prints a local URL like:

```text
http://127.0.0.1:8765
```

Open that in a browser.

## What The GUI Shows

### FEN And Prompt Controls

- `FEN` sets the chess position to inspect
- `Prompt` sets the user text prompt before generation begins
- `Restart` rebuilds the decode session from that FEN and prompt

When a session starts, the model runs one full forward pass over the prompt and
position context. At that point no generated token has been committed yet, but
the GUI already shows the current next-token distribution.

### Generated Text And Step Controls

- `Generated text` shows the tokens that have already been emitted
- `Step Back` rewinds to the previous decode boundary and restores the earlier top-5 distribution
- `Step (Greedy)` appends the current argmax token
- clicking one of the top-5 token buttons forces that token instead

Each forward step does exactly one token of decoding and then recomputes the
next-token distribution from the updated cache. `Step Back` restores the prior
decode state by replaying the session up to the previous emitted-token prefix.

### Top-5 Next Tokens

The top-token panel shows the current softmax probabilities over the next token:

- probabilities are computed from the current next-token logits
- the display is always the top 5 entries
- these are raw decode probabilities at temperature `1.0`

This view is about what the model will do next, not what it did previously.

### Layer Selector

The layer dropdown chooses which fusion layer's routing to visualize.

The boards always correspond to the selected layer's routing for the last valid
text token processed in that decode step.

## Interpreting The Heatmaps

The GUI shows four 64-square boards:

- `Aggregate`
- `CSMP`
- `Perceiver`
- `Policy`

Squares are rendered in normal chess orientation with White at the bottom.
Internally, square index `0` is `a1`, matching `python-chess`.

### Aggregate Board

The aggregate board is the source-marginalized 64-square distribution for the
last token:

$$
p_i^{sq} = \sum_s \alpha_{s,i}
$$

This is the easiest high-level answer to "which board squares mattered most for
this token?"

### Source Boards

The three source boards split that aggregate mass by aligned source type:

- `CSMP`: routing to the message-passing square tokens
- `Perceiver`: routing to the Perceiver square latents
- `Policy`: routing to the structured policy latents

For a selected square, these three values sum to that square's aggregate mass.

The source order is fixed everywhere in the inspector:

1. `CSMP`
2. `Perceiver`
3. `Policy`

### Raw Slot Weights

Under the hood, the structured mixer routes over `192` aligned slots:

- `64` CSMP square slots
- `64` Perceiver square slots
- `64` Policy square slots

The GUI's board views are derived from those raw `192` values. Tooltips and raw
value labels are meant to make that relationship inspectable without forcing you
to look at the tensor directly.

## How One Decode Step Maps To The Model

For each decode step:

1. The prompt, FEN features, prefix embeddings, masks, and chess context are prepared.
2. The model computes the next-token logits.
3. The selected token is committed, either greedily or from a clicked top-5 choice.
4. The inspector captures the chosen fusion layer traces for the last valid token only.
5. The UI updates the text, top-5 distribution, and square heatmaps.

Only the current step's last-token trace is stored. The inspector does not keep
a full history of all layer traces across the entire generation.

## What The GUI Is Not Showing

A few boundaries are important:

- it is not a training dashboard
- it does not show all tokens in the sequence at once
- it does not visualize every internal tensor in the fusion layer
- it does not currently expose stochastic sampling controls

The square boards answer "what did this layer use for the last token?" not
"what was the layer doing over the whole sentence?"

## Relationship To The Math Document

The inspector is the visual counterpart to
[structured_square_mixer_math.md](structured_square_mixer_math.md):

- top-5 next tokens come from the current decoder logits
- `CSMP`, `Perceiver`, and `Policy` boards correspond to the three aligned slot groups
- `Aggregate` corresponds to the source-marginalized 64-square distribution
- the per-layer view lets you compare how routing changes across decoder depth

If you want the exact equations, use the math doc. If you want to see one decode
step play out visually on a real position, use the inspector.

## Troubleshooting

- If the GUI shows an API error, check the terminal running `decoding_inspector.py`.
- Server-side failures are printed there with a Python traceback.
- Invalid FEN strings or unsupported checkpoints will fail when a session starts.
