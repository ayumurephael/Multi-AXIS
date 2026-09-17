# Multi-AXIS: Explaining Multivariate Time-Series Anomalies with Detector-Derived Contextual Evidence

Multi-AXIS connects a frozen multivariate time-series anomaly detector to a
frozen language or vision-language model through a compact trainable Hint
Tuner. The detector supplies time- and channel-aligned contextual evidence;
the Hint Tuner converts that evidence into soft prompt embeddings; and the
language model produces a structured anomaly decision and explanation.

This repository contains the Multi-AXIS method implementation.
Experiment-specific hyperparameters, distributed launch profiles, historical
predictions, and G-Eval artifacts are not part of the public repository.

## Method overview

```text
multivariate series X and target interval [s,e)
        │
        ├── deterministic per-channel normalization and numeric Window
        ├── optional normalized full-series figure
        │
        ▼
frozen TimeRCD
        ├── local representation h(t,c)
        └── anomaly logits a(t,c)
        │
        ▼
detector-derived evidence enhancement
        │
        ├── Step-Local hints: one hint per target time step and channel
        ├── Channel-Local hints: one summary per channel
        ├── Joint-Local hint: question-conditioned multivariate summary
        └── Fixed hints: task-level priors
        │
        ▼
native LLM/VLM chat prompt with soft-hint placeholders
        │
        ▼
MC / TF / OE response contract and evidence-based explanation
```

The LLM/VLM and TimeRCD parameters remain frozen. The trainable bridge contains
the prototype mapping, anomaly direction, projections, pooling queries, and
cross-attention layers used to form the four hint families. During supervised
forward propagation, loss is computed only on answer tokens.

See [architecture.md](docs/architecture.md) for equations and tensor shapes,
[method_flow.md](docs/method_flow.md) for the end-to-end execution path, and
[interfaces.md](docs/interfaces.md) for record and module contracts.

## Repository structure

```text
.
├── src/multi_axis/             # Multi-AXIS model and runtime implementation
│   ├── timeseries_encoder.py   # checkpoint-compatible TimeRCD encoder
│   ├── timercd.py              # frozen detector wrapper and pretraining heads
│   ├── hint_tuner.py           # detector evidence → four soft-hint families
│   ├── attention.py            # FlashAttention-2/reference cross-attention
│   ├── model.py                # frozen LLM/VLM integration and answer-only loss
│   ├── prompting.py            # numeric/visual prompt and hint placeholders
│   ├── data.py                 # normalization, manifest dataset, batching
│   ├── rendering.py            # deterministic normalized-series figures
│   ├── inference.py            # generic ordered inference loop
│   └── response_contracts.py   # MC/TF/OE parsing and target normalization
├── scripts/
│   ├── prepare_data.py         # question JSONL → random-access manifest
│   ├── render_series.py        # offline VLM figure generation
│   └── run_inference.py        # single-device end-to-end inference
├── configs/                    # method/runtime configuration only
├── examples/                   # lightweight structural example
├── tests/                      # CPU tests for core contracts and tensor shapes
└── docs/                       # architecture, flow and interface documentation
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

Formal CUDA runs require a FlashAttention-2 build compatible with the target
PyTorch/CUDA environment. CPU execution uses PyTorch scaled dot-product
attention and is intended for structural tests, not full-model throughput.

## Preparing model input

The final QA JSONL must retain `series.values`, `target_interval`, `question`,
the question type, channel metadata, and a stable `sample_id`. When supervised
answers are needed, it must also contain one of the supported answer fields.

```bash
python scripts/prepare_data.py \
  --data-root /path/to/data \
  --inputs /path/to/data/questions_and_answers.jsonl \
  --output /path/to/manifests/data.jsonl \
  --require-answer
```

The command writes the manifest plus `.idx.json` and `.groups.json` sidecars.
Rows that share a `base_sample_id` stay grouped so their frozen TimeRCD output
can be reused without mixing question-specific evidence.

For VLM execution, render the complete normalized sequence before inference:

```bash
python scripts/render_series.py \
  --config configs/multi_axis_vl.json \
  --data-root /path/to/data \
  --manifest-dir /path/to/manifests \
  --output-root /path/to/rendered_images \
  --manifests data.jsonl
```

The renderer consumes exactly the per-channel-normalized model input. It marks
the half-open target interval `[start,end)` without drawing anomaly labels,
scores, severity, root cause, or causal direction.

## Inference

Inference needs the base LLM/VLM, a TimeRCD checkpoint, a trained Hint Tuner
checkpoint, the prepared manifest, and a writable question-semantic cache.
Model weights and checkpoints are not included.

```bash
python scripts/run_inference.py \
  --config configs/multi_axis_vl.json \
  --manifest /path/to/manifests/data.jsonl \
  --data-root /path/to/data \
  --image-root /path/to/rendered_images \
  --timercd-checkpoint /path/to/timercd.pt \
  --hint-checkpoint /path/to/multi_axis_hints.pt \
  --question-cache /path/to/question_cache \
  --output /path/to/predictions.jsonl
```

Each prediction records the raw response and an output-contract error code, if
the response violates its MC, TF, or OE format.

## Lightweight verification

The structural example does not download a language model or claim meaningful
anomaly predictions. It verifies the TimeRCD-to-hint tensor path with small
random tensors.

```bash
python examples/minimal_pipeline.py
pytest
```

## License

Released under the [MIT License](LICENSE).

