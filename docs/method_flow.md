# End-to-end method flow

## A. Supply QA records

Provide existing JSONL records containing the complete time series, channel
metadata, target interval, question, and an optional reference answer for
supervision. See [interfaces.md](interfaces.md) for the input contract. The
model-facing prompt receives observations and learned detector evidence, not
reference answers or ground-truth anomaly labels.

## B. Build a random-access manifest

`scripts/prepare_data.py` validates `series.values` and `target_interval`, maps
the question type to MC, TF, or OE, records the byte range of the authoritative
JSONL row, and writes three files:

- `<name>.jsonl`: compact records used for identity and lookup;
- `<name>.jsonl.idx.json`: byte offsets for constant-time manifest access;
- `<name>.jsonl.groups.json`: row indices grouped by `base_sample_id`.

The source JSONL remains authoritative. `ManifestDataset` checks that question
text and interval still match the manifest, so silently replacing the source
file cannot change model input unnoticed.

## C. Materialize one model sample

For each row, `ManifestDataset`:

1. decodes the full finite `T × C` sequence;
2. validates the half-open target interval;
3. normalizes every channel over its complete valid sequence;
4. serializes the target interval in channel-major order as rounded integers;
5. canonicalizes all channel identifiers as `ch_<index>`;
6. resolves the deterministic image identity and path;
7. optionally canonicalizes the teacher answer to its response contract.

`collate_multiaxis` pads time and channel dimensions separately and emits
`time_mask` and `channel_mask`, preventing padded positions or absent channels
from becoming valid evidence.

## D. Render the optional visual view

`scripts/render_series.py` reads the same `ManifestDataset.normalized_series`
consumed by the detector; it does not normalize a second time. Each channel is
drawn on a separate row. The complete context is black, the target segment is
dark blue, and the target background is pale yellow. Channel labels, ordering,
and global indices match the numeric prompt. A render manifest binds the image
to the normalized-series hash, base sample, interval, renderer version, font,
and file hash.

## E. Produce detector-derived hints

The frozen TimeRCD encoder processes the padded normalized series once per base
sample. For each question sharing that base sample, the model selects the
question's interval, combines local representations with anomaly-logit
evidence, and forms Step-Local, Channel-Local, and Joint-Local hints. Fixed hints
are formed from learned task queries. The prototype bank can be computed once
per inference run because it depends on the frozen LLM embeddings and trained
prototype mapping, not the sample.

## F. Assemble the native LLM/VLM input

The prompt builder creates system and user messages through the checkpoint's
native chat template. The processor then produces text tokens and, when enabled,
native visual tensors. The builder verifies:

- every hint token is atomic;
- the expected number of each placeholder occurs;
- the prompt is an exact prefix of the supervised sequence;
- no context truncation occurs;
- every visual input contains pixel values, image grid, and multimodal token
  type IDs;
- no local image path leaks into the text.

Only after these checks does the embedding hook replace placeholder embeddings.

## G. Train or infer

The public `forward` method supports answer-only supervised loss and preserves
the exact freeze contract. `scripts/run_inference.py` loads TimeRCD and Hint
Tuner checkpoints, fills or reads the question-semantic cache, reuses TimeRCD
output for adjacent questions with the same `base_sample_id`, and generates one
visible answer per row.

## H. Parse the response

`response_contracts.py` performs strict first-line parsing for MC/TF and strict
section parsing for OE. It returns an explicit contract error rather than
searching later prose for a convenient label. Prediction JSONL therefore keeps
both the raw answer and its parse status.

