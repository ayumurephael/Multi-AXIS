from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .channel_ids import (
    canonical_channel_ids,
    canonicalize_question_channel_references,
)
from .response_contracts import canonicalize_teacher_answer, response_contract_error


TEACHER_ANSWER_FIELDS = ("model_answer", "teacher_answer_llm", "windows_0_answer", "answer")
IMAGE_LAYOUT_VERSION = "multi-axis-vl-render-v1"


def teacher_model_answer(row: Dict[str, Any]) -> str:
    """Return only the natural-language model_answer used for formal supervision."""
    value = row.get("model_answer")
    return value.strip() if isinstance(value, str) else ""


def teacher_answer(row: Dict[str, Any]) -> str:
    """Best-effort answer extraction for evaluation/reference compatibility."""
    for field in TEACHER_ANSWER_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def question_type_group(row: Dict[str, Any]) -> str:
    value = str(row.get("question_answer_type", "")).strip().lower()
    mapping = {
        "choice": "MC",
        "multiple_choice": "MC",
        "mc": "MC",
        "open": "OE",
        "open_ended": "OE",
        "oe": "OE",
        "judgment": "TF",
        "true_false": "TF",
        "tf": "TF",
    }
    if value in mapping:
        return mapping[value]
    question_type = str(row.get("question_type", "")).lower()
    if "choice" in question_type:
        return "MC"
    if "judg" in question_type or "true_false" in question_type:
        return "TF"
    if "open" in question_type:
        return "OE"
    raise ValueError(f"Cannot map question type: {value!r} / {question_type!r}")


def extract_values(row: Dict[str, Any]) -> np.ndarray:
    series = row.get("series")
    if not isinstance(series, dict) or "values" not in series:
        raise ValueError("Question record lacks series.values")
    values = np.asarray(series["values"], dtype=np.float32)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError(f"series.values must be non-empty [T,C], got {values.shape}")
    declared = series.get("shape")
    if declared is not None and tuple(declared) != tuple(values.shape):
        raise ValueError(f"Declared series shape {declared} != decoded shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("series.values contains NaN or infinity")
    return values


def extract_interval(row: Dict[str, Any], steps: int) -> Tuple[int, int]:
    interval = row.get("target_interval")
    if not isinstance(interval, dict):
        raise ValueError("Question record lacks target_interval")
    start, end = int(interval["start"]), int(interval["end"])
    if not 0 <= start < end <= steps:
        raise ValueError(f"Invalid target interval [{start}, {end}) for T={steps}")
    return start, end


def extract_channel_ids(row: Dict[str, Any], channels: int) -> List[str]:
    metadata = row.get("channels") or []
    raw_identifiers: List[Any] = []
    for index in range(channels):
        item = metadata[index] if index < len(metadata) else None
        if isinstance(item, dict):
            value = item.get("channel_id") or item.get("name")
        else:
            value = item
        raw_identifiers.append(value)
    return canonical_channel_ids(raw_identifiers, channels)


def structured_is_anomalous(row: Dict[str, Any]) -> bool | None:
    candidates = [
        row.get("is_anomalous"),
        (row.get("target_output") or {}).get("is_anomalous"),
        ((row.get("target_output") or {}).get("fact_check") or {}).get("is_anomalous"),
    ]
    for value in candidates:
        if isinstance(value, bool):
            return value
    return None


def visual_image_id(
    base_sample_id: str,
    interval: Tuple[int, int],
    renderer_version: str = IMAGE_LAYOUT_VERSION,
) -> str:
    start, end = interval
    payload = f"{renderer_version}\0{base_sample_id}\0{start}\0{end}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def visual_image_relpath(image_id: str) -> str:
    if len(image_id) != 64 or any(char not in "0123456789abcdef" for char in image_id):
        raise ValueError("Visual image id must be a lowercase SHA-256 digest")
    return f"images/{image_id[:2]}/{image_id}.png"


def normalize_and_serialize(
    values: np.ndarray,
    interval: Tuple[int, int],
    epsilon: float = 1e-5,
    scale: int = 100,
) -> Tuple[np.ndarray, List[List[int]], List[float], List[float]]:
    mean = values.mean(axis=0, keepdims=True, dtype=np.float64)
    std = values.std(axis=0, keepdims=True, dtype=np.float64)
    normalized = ((values.astype(np.float64) - mean) / (std + epsilon)).astype(np.float32)
    start, end = interval
    serialized = np.rint(normalized[start:end].T * scale).astype(np.int64).tolist()
    if len(serialized) != values.shape[1] or any(len(x) != end - start for x in serialized):
        raise AssertionError("Channel-major serialized window shape is inconsistent")
    return (
        normalized,
        serialized,
        mean.reshape(-1).astype(np.float64).tolist(),
        std.reshape(-1).astype(np.float64).tolist(),
    )


class ManifestDataset(Dataset):
    """Random-access sanitized manifest; question JSONL is read by byte offset."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        window_epsilon: float = 1e-5,
        window_scale: int = 100,
        image_root: str | Path | None = None,
        require_image: bool = False,
        renderer_version: str = IMAGE_LAYOUT_VERSION,
        normalize_teacher_targets: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.window_epsilon = window_epsilon
        self.window_scale = window_scale
        self.image_root = Path(image_root) if image_root is not None else None
        self.require_image = bool(require_image)
        self.renderer_version = str(renderer_version)
        self.normalize_teacher_targets = bool(normalize_teacher_targets)
        index_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".idx.json")
        self.offsets = json.loads(index_path.read_text(encoding="utf-8"))
        groups_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".groups.json")
        self.groups = (
            json.loads(groups_path.read_text(encoding="utf-8"))
            if groups_path.exists() else [[index] for index in range(len(self.offsets))]
        )
        self._manifest_handle = None
        self._question_handles: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_manifest_handle"] = None
        state["_question_handles"] = {}
        return state

    def _manifest_record(self, index: int) -> Dict[str, Any]:
        if self._manifest_handle is None:
            self._manifest_handle = self.manifest_path.open("rb")
        self._manifest_handle.seek(self.offsets[index])
        return json.loads(self._manifest_handle.readline())

    def _question_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        relative = record["question_path"]
        handle = self._question_handles.get(relative)
        if handle is None:
            handle = (self.data_root / relative).open("rb")
            self._question_handles[relative] = handle
        handle.seek(record["question_offset"])
        raw = handle.read(record["question_length"])
        row = json.loads(raw)
        if str(row.get("question")) != record["question"]:
            raise RuntimeError("Manifest/question text mismatch; source data changed")
        return row

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self._manifest_record(index)
        row = self._question_record(record)
        values = extract_values(row)
        interval = extract_interval(row, values.shape[0])
        if list(interval) != record["interval"]:
            raise RuntimeError("Manifest/question interval mismatch; source data changed")
        normalized, serialized, channel_means, channel_stds = normalize_and_serialize(
            values,
            interval,
            epsilon=self.window_epsilon,
            scale=self.window_scale,
        )
        channel_ids = extract_channel_ids(row, values.shape[1])
        image_id = str(
            record.get("image_id")
            or visual_image_id(record["base_sample_id"], interval, self.renderer_version)
        )
        image_relpath = str(record.get("image_relpath") or visual_image_relpath(image_id))
        image_path = self.image_root / Path(image_relpath) if self.image_root is not None else None
        if self.require_image and (image_path is None or not image_path.is_file()):
            raise FileNotFoundError(
                f"Missing pre-rendered VLM image for sample {record['sample_id']}: {image_path}"
            )
        answer = record.get("teacher_answer")
        answer_was_canonicalized = False
        if self.normalize_teacher_targets and answer:
            canonical = canonicalize_teacher_answer(
                answer,
                record["question_group"],
                structured_is_anomalous=structured_is_anomalous(row),
            )
            answer_was_canonicalized = canonical != str(answer).strip()
            answer = canonical
            if response_contract_error(answer, record["question_group"]) is not None:
                raise AssertionError("Canonicalized teacher answer violates its output contract")
        question = canonicalize_question_channel_references(record["question"])
        sample = {
            "normalized_series": normalized,
            "question": question,
            "question_was_canonicalized": question != record["question"],
            "answer": answer,
            "answer_was_canonicalized": answer_was_canonicalized,
            "interval": interval,
            "window_values": serialized,
            "channel_means": channel_means,
            "channel_stds": channel_stds,
            "time_count": values.shape[0],
            "channel_count": values.shape[1],
            "channel_ids": channel_ids,
            "sample_id": record["sample_id"],
            "base_sample_id": record["base_sample_id"],
            "question_group": record["question_group"],
            "dataset": record.get("dataset", "train"),
            "image_id": image_id,
            "image_relpath": image_relpath,
            "image_path": str(image_path) if image_path is not None else None,
        }
        if "label_reference" in record:
            sample["label_reference"] = record["label_reference"]
        return sample


def collate_multiaxis(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    max_time = max(sample["time_count"] for sample in samples)
    max_channels = max(sample["channel_count"] for sample in samples)
    batch = len(samples)
    series = torch.zeros(batch, max_time, max_channels, dtype=torch.float32)
    time_mask = torch.zeros(batch, max_time, dtype=torch.bool)
    channel_mask = torch.zeros(batch, max_channels, dtype=torch.bool)
    for index, sample in enumerate(samples):
        steps, channels = sample["time_count"], sample["channel_count"]
        value = torch.from_numpy(sample["normalized_series"])
        if value.shape != (steps, channels):
            raise AssertionError("Normalized series shape changed during collation")
        series[index, :steps, :channels] = value
        time_mask[index, :steps] = True
        channel_mask[index, :channels] = True
    return {
        "normalized_series": series,
        "time_mask": time_mask,
        "channel_mask": channel_mask,
        "questions": [sample["question"] for sample in samples],
        "answers": [sample.get("answer") for sample in samples],
        "intervals": [sample["interval"] for sample in samples],
        "window_values": [sample["window_values"] for sample in samples],
        "channel_means": [sample["channel_means"] for sample in samples],
        "channel_stds": [sample["channel_stds"] for sample in samples],
        "time_counts": [sample["time_count"] for sample in samples],
        "channel_counts": [sample["channel_count"] for sample in samples],
        "channel_ids": [sample["channel_ids"] for sample in samples],
        "sample_ids": [sample["sample_id"] for sample in samples],
        "base_sample_ids": [sample["base_sample_id"] for sample in samples],
        "question_groups": [sample["question_group"] for sample in samples],
        "datasets": [sample["dataset"] for sample in samples],
        "label_references": [sample.get("label_reference") for sample in samples],
        "image_ids": [sample["image_id"] for sample in samples],
        "image_paths": [sample.get("image_path") for sample in samples],
        "answer_was_canonicalized": [
            sample.get("answer_was_canonicalized", False) for sample in samples
        ],
        "question_was_canonicalized": [
            sample.get("question_was_canonicalized", False) for sample in samples
        ],
    }
