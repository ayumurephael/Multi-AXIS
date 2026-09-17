from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from .data import ManifestDataset, collate_multiaxis
from .response_contracts import response_contract_error


def model_inputs_from_batch(
    batch: Mapping[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Select and transfer the fields consumed by ``generate_answers``."""

    return {
        "normalized_series": batch["normalized_series"].to(device),
        "time_mask": batch["time_mask"].to(device),
        "channel_mask": batch["channel_mask"].to(device),
        "questions": batch["questions"],
        "intervals": batch["intervals"],
        "channel_counts": batch["channel_counts"],
        "window_values": batch["window_values"],
        "channel_means": batch["channel_means"],
        "channel_stds": batch["channel_stds"],
        "channel_ids": batch["channel_ids"],
        "question_groups": batch["question_groups"],
        "image_paths": batch["image_paths"],
    }


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def predict_dataset(
    model,
    dataset: ManifestDataset,
    device: torch.device,
    *,
    output_path: Optional[str | Path] = None,
    generation_overrides: Optional[Mapping[str, Any]] = None,
) -> list[Dict[str, Any]]:
    """Run the complete Multi-AXIS inference path over a prepared manifest.

    Samples are processed in manifest order. Adjacent questions sharing one
    ``base_sample_id`` reuse the frozen TimeRCD output while retaining their own
    question-conditioned Joint-Local hint and output contract.
    """

    model.eval()
    model.timercd.to(device)
    model.hint_tuner.to(device)
    records: list[Dict[str, Any]] = []
    cached_base_sample_id: Optional[str] = None
    cached_timercd = None
    autocast = (
        torch.autocast("cuda", dtype=getattr(torch, model.config.llm.torch_dtype))
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        prototype = model.build_prototype_bank()
        for index in range(len(dataset)):
            sample = dataset[index]
            batch = collate_multiaxis([sample])
            inputs = model_inputs_from_batch(batch, device)
            if cached_base_sample_id != sample["base_sample_id"]:
                cached_timercd = model.timercd(
                    inputs["normalized_series"],
                    inputs["time_mask"],
                    inputs["channel_mask"],
                )
                cached_base_sample_id = sample["base_sample_id"]
            started = time.monotonic()
            response = model.generate_answers(
                **inputs,
                prototype_override=prototype,
                timercd_override=cached_timercd,
                generation_overrides=generation_overrides,
            )[0]
            records.append(
                {
                    "index": index,
                    "sample_id": sample["sample_id"],
                    "base_sample_id": sample["base_sample_id"],
                    "question_group": sample["question_group"],
                    "raw_response": response,
                    "contract_error": response_contract_error(
                        response, sample["question_group"]
                    ),
                    "generation_seconds": time.monotonic() - started,
                    "model": model.config.llm.model_name,
                    "image_id": sample["image_id"] if model.config.vision.enabled else None,
                }
            )
            if output_path is not None:
                _atomic_write_jsonl(Path(output_path), records)
    return records
