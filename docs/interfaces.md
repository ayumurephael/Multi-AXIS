# Core interfaces

## Question JSONL record

Required model-facing fields:

| Field | Contract |
|---|---|
| `sample_id` | Stable identity for one question/frame record. |
| `base_sample_id` | Stable identity shared by questions using the same series and interval; defaults to `sample_id`. |
| `series.values` | Finite nested list with shape `[T,C]`. |
| `series.shape` | Optional declared shape; when present it must equal the decoded shape. |
| `target_interval.start/end` | Valid half-open interval `0 <= start < end <= T`. |
| `question` | Non-empty question; numeric channel references are canonicalized to `ch_<index>`. |
| `question_answer_type` | Choice, judgment, or open form mapped to `MC`, `TF`, or `OE`. |
| `channels` | Optional length-`C` channel descriptors containing numeric channel IDs or names. |
| `model_answer` | Preferred natural-language teacher answer for supervision. |

`teacher_answer_llm`, `windows_0_answer`, and `answer` are accepted as fallback
reference fields. Formal supervision should use `model_answer` whenever it is
available in the input record.

## `ManifestDataset` output

Important fields and shapes for one sample:

| Field | Type/shape | Meaning |
|---|---|---|
| `normalized_series` | NumPy `[T,C]` | Full per-channel-normalized sequence. |
| `interval` | `(start,end)` | Half-open target interval. |
| `window_values` | list `[C,e-s]` | Channel-major normalized integers. |
| `channel_means/stds` | list `[C]` | Full-series raw-scale statistics. |
| `channel_ids` | list `[C]` | Canonical `ch_<index>` identifiers. |
| `question_group` | `MC`, `TF`, or `OE` | Selects the response contract. |
| `image_path` | optional path | Deterministic pre-rendered visual input. |

`collate_multiaxis` pads these samples into `[B,T_max,C_max]` and returns
separate Boolean time and channel masks.

## `FrozenTimeRCD.forward`

Inputs:

- `normalized_series`: float tensor `[B,T,C]`;
- `time_mask`: Boolean tensor `[B,T]`;
- `channel_mask`: Boolean tensor `[B,C]`.

Outputs:

- local representations `[B,T,C,d_proj]`;
- per-position/channel anomaly logits `[B,T,C,2]`.

The wrapper always remains in evaluation mode and executes its backbone under
`torch.no_grad()`.

## `MultiAxisHintTuner.forward`

In addition to TimeRCD outputs, the tuner receives one interval and channel
count per sample, detached question semantics `[B,d_L]`, the frozen vocabulary
embedding matrix `[V,d_L]`, and an optional precomputed prototype bank.

It returns:

- a list of Step-Local tensors `[C_i(e_i-s_i),d_L]`;
- a list of Channel-Local tensors `[C_i,d_L]`;
- a Joint-Local tensor `[B,1,d_L]`;
- a Fixed tensor `[B,30,d_L]` in the complete configuration.

## `MultiAxisForConditionalGeneration.forward`

This supervised entry point accepts the collated tensors plus questions,
answers, intervals, serialized windows, statistics, channel IDs, question
groups, and optional image paths. It returns `AnswerOnlyCausalLMOutput` with the
mean answer-token NLL, supervised-token count, and visual-token count.

## `generate_answers`

The inference entry point accepts the same evidence fields without answers. It
supports optional precomputed TimeRCD output and prototype bank, which avoids
repeating frozen computation across related questions. It returns one decoded
string per batch item.

## Checkpoint expectations

TimeRCD loading accepts a dictionary under `model_state_dict`, `state_dict`, or
the top-level payload and uses strict parameter matching by default. The
publicly extracted backbone preserves the original `ts_encoder`,
`reconstruction_head`, and `anomaly_head` parameter paths.

Hint checkpoints contain only `hint_tuner_state_dict` plus format, configuration,
TimeRCD hash, training position, and metadata. Loading rejects an incompatible
format or a TimeRCD hash mismatch.

