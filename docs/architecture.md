# Multi-AXIS architecture

## 1. Inputs and normalization

Let a sample be a multivariate sequence

\[
X=[x_{t,c}]\in\mathbb{R}^{T\times C},
\]

with a half-open target interval \([s,e)\). Each channel is normalized over
its complete valid sequence:

\[
\mu_c=\frac{1}{T}\sum_{t=1}^{T}x_{t,c},\qquad
\sigma_c=\sqrt{\frac{1}{T}\sum_{t=1}^{T}(x_{t,c}-\mu_c)^2},
\]

\[
\bar{x}_{t,c}=\frac{x_{t,c}-\mu_c}{\sigma_c+\epsilon_w}.
\]

The prompt's numeric Window stores
\(v_{t,c}=\operatorname{round}(100\bar{x}_{t,c})\) for
\(t\in[s,e)\), together with \(\mu_c\), \(\sigma_c\), and
\(\epsilon_w=10^{-5}\). Therefore the approximate raw value is recoverable as

\[
x_{t,c}\approx\mu_c+(\sigma_c+\epsilon_w)\frac{v_{t,c}}{100}.
\]

The optional figure and numeric Window are two representations of the same
observations. The figure contains the complete normalized context; the Window
contains exact channel-aligned values for the target interval.

## 2. Frozen TimeRCD evidence source

TimeRCD divides each channel into temporal patches, projects every patch into a
latent space, and applies a non-causal Transformer encoder with rotary temporal
position embeddings. For multivariate input, a learned binary attention bias
distinguishes token pairs from the same and different channels.

The checkpoint-compatible backbone returns

\[
H=\{h_{t,c}\}\in\mathbb{R}^{B\times T\times C\times d_p}
\]

and applies a frozen anomaly head to obtain

\[
A=\{a_{t,c,0},a_{t,c,1}\}
\in\mathbb{R}^{B\times T\times C\times 2}.
\]

Both the encoder and anomaly head are frozen. The anomaly logits are treated as
detector-derived evidence, not calibrated probabilities or ground-truth labels.

## 3. Evidence enhancement

For each target position and channel, the anomaly evidence scalar is

\[
z_{t,c}=\tanh\left(\frac{a_{t,c,1}-a_{t,c,0}}{2}\right).
\]

Let \(d_a\in\mathbb{R}^{d_p}\) be a trainable anomaly direction. The enhanced
local representation is

\[
\tilde h_{t,c}=\operatorname{RMS}
\left(\operatorname{RMS}(h_{t,c})+z_{t,c}d_a\right).
\]

This operation retains the detector's contextual representation while giving
its anomaly evidence a learnable direction in the same space.

## 4. Vocabulary-derived prototype bank

Let \(E\in\mathbb{R}^{V\times d_L}\) be the frozen LLM input embedding matrix.
A trainable mapping \(M\in\mathbb{R}^{N_p\times V}\) constructs

\[
P=ME\in\mathbb{R}^{N_p\times d_L},
\]

where the complete method uses \(N_p=1024\). All four hint families attend to
this shared vocabulary-derived prototype bank, keeping their outputs in the
LLM embedding space.

## 5. Four contextual hint families

### Step-Local

The target tensor is flattened in channel-major order. Each
\(\tilde h_{t,c}\) is projected to \(d_L\) and cross-attends to \(P\), producing
one Step-Local hint per \((t,c)\). Its prompt placeholder is placed immediately
after the numeric value for the same channel and global time index.

### Channel-Local

For each channel, a shared trainable query pools all enhanced representations
in \([s,e)\). The pooled vector is projected to \(d_L\) and attends to \(P\),
producing one question-independent summary per channel. Pooling may reflect
cross-channel context already encoded by TimeRCD, but its output remains
anchored to one named channel.

### Joint-Local

The frozen language backbone encodes the exact question and returns its final
valid hidden state \(h_q\). A trainable projection conditions a shared joint
query:

\[
q_J=q_0+W_q\operatorname{RMS}(h_q).
\]

This query pools all enhanced target positions and channels, after which the
pooled vector attends to \(P\). The result is one question-conditioned summary
of the complete multivariate target window. Question states are detached and
stored in an identity-validated disk cache.

### Fixed

Thirty learned queries attend to \(P\) and provide task-level priors. They are
not sample-specific facts.

## 6. Native prompt integration

The prompt builder creates atomic placeholders for Fixed, Step-Local,
Channel-Local, and Joint-Local hints. It also includes:

- the exact question and MC/TF/OE output contract;
- the target interval and reconstruction equation;
- per-channel means, standard deviations, and normalized integer values;
- canonical `ch_<index>` identifiers;
- optionally, the full normalized sequence figure.

After the tokenizer/processor builds native model inputs, a forward hook on the
LLM input embedding layer replaces only the validated placeholder positions
with the computed soft hints. The native Qwen3-VL image tokens, pixel tensors,
image grid, multimodal token types, M-RoPE, and generation path remain intact.

## 7. Supervision and generation

Teacher forcing masks the entire prompt with label `-100`; only assistant answer
tokens contribute to causal negative log-likelihood. The implementation selects
the hidden states aligned with supervised next tokens before applying the output
embedding, avoiding vocabulary logits for prompt and visual positions.

At generation time, placeholder and hidden-reasoning markers are forbidden from
the visible output. MC responses must start with `Answer: A` through `Answer: D`;
TF responses must start with `Yes.` or `No.`; OE responses must contain the
ordered `Decision`, `Main evidence`, and `Interpretation` sections.

## 8. Trainable and frozen parameters

Frozen components:

- TimeRCD encoder, reconstruction head, and anomaly head;
- LLM/VLM, token embeddings, visual encoder, and output head;
- cached question semantic states.

Trainable Hint Tuner components:

- prototype mapping;
- anomaly direction;
- local-to-LLM projection;
- prototype, channel, and joint cross-attention layers;
- channel, joint, and fixed queries;
- question projection.

