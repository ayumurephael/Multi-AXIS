from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .config import MultiAxisConfig
from .data import (
    ManifestDataset,
    visual_image_id,
    visual_image_relpath,
)


CONTEXT_COLOR = "#000000"
TARGET_COLOR = "#173f7a"
TARGET_BACKGROUND = "#fff2b2"
RENDER_FORMAT = "multi-axis-vl-image-manifest-v1"
TIMES_FONT_FILES = ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_series_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def require_times_new_roman() -> Dict[str, Any]:
    from matplotlib import font_manager

    explicit_root = os.environ.get("MULTI_AXIS_TIMES_FONT_DIR")
    registered: List[Path] = []
    if explicit_root:
        font_root = Path(explicit_root).resolve()
        if not font_root.is_dir():
            raise RuntimeError(
                f"MULTI_AXIS_TIMES_FONT_DIR is not a directory: {font_root}"
            )
        missing = [name for name in TIMES_FONT_FILES if not (font_root / name).is_file()]
        if missing:
            raise RuntimeError(
                f"The explicit Times New Roman bundle is incomplete: {missing}"
            )
        for name in TIMES_FONT_FILES:
            path = font_root / name
            font_manager.fontManager.addfont(str(path))
            registered.append(path)
    try:
        resolved = Path(
            font_manager.findfont("Times New Roman", fallback_to_default=False)
        ).resolve()
    except ValueError as exc:
        raise RuntimeError(
            "Times New Roman is required by the VLM rendering specification but "
            "is not installed on this compute node."
        ) from exc
    if font_manager.FontProperties(fname=str(resolved)).get_name() != "Times New Roman":
        raise RuntimeError("Resolved renderer font is not Times New Roman")
    audited = registered or [resolved]
    files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in audited
    ]
    bundle_digest = hashlib.sha256()
    for item in files:
        bundle_digest.update(item["name"].encode("utf-8"))
        bundle_digest.update(b"\0")
        bundle_digest.update(item["sha256"].encode("ascii"))
        bundle_digest.update(b"\n")
    return {
        "family": "Times New Roman",
        "registration_source": "explicit_bundle" if registered else "system_font",
        "bundle_sha256": bundle_digest.hexdigest(),
        "files": files,
    }


def render_normalized_series(
    normalized_series: np.ndarray,
    interval: Tuple[int, int],
    channel_ids: Sequence[str],
    output_path: Path,
    *,
    dpi: int = 600,
) -> Dict[str, Any]:
    """Render exactly the normalized model input; no normalization occurs here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(normalized_series, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("normalized_series must be finite [T,C]")
    steps, channels = values.shape
    start, end = map(int, interval)
    if not 0 <= start < end <= steps:
        raise ValueError(f"Invalid half-open interval [{start}, {end}) for T={steps}")
    if len(channel_ids) != channels:
        raise ValueError("Channel label count does not match normalized series")
    if dpi != 600:
        raise ValueError("The VLM architecture fixes renderer dpi at 600")

    font_audit = require_times_new_roman()
    figure_height = max(2.4, min(16.0, 1.0 + 0.55 * channels))
    font_size = max(4.0, min(7.0, 8.0 - 0.06 * channels))
    x = np.arange(steps, dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.png")
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": font_size,
        "axes.unicode_minus": False,
        "savefig.dpi": dpi,
        "figure.dpi": dpi,
    }
    with plt.rc_context(rc):
        figure, axes = plt.subplots(
            channels,
            1,
            figsize=(6.4, figure_height),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        try:
            for channel, axis in enumerate(axes):
                axis.axvspan(
                    start - 0.5,
                    end - 0.5,
                    facecolor=TARGET_BACKGROUND,
                    edgecolor="none",
                    zorder=0,
                )
                axis.plot(
                    x,
                    values[:, channel],
                    color=CONTEXT_COLOR,
                    linewidth=0.55,
                    zorder=1,
                )
                axis.plot(
                    x[start:end],
                    values[start:end, channel],
                    color=TARGET_COLOR,
                    linewidth=1.05,
                    zorder=2,
                )
                axis.axhline(0.0, color="#4b5563", linewidth=0.25, zorder=0)
                axis.set_ylabel(str(channel_ids[channel]), rotation=0, ha="right", va="center")
                axis.set_xlim(-0.5, steps - 0.5)
                axis.tick_params(axis="x", labelbottom=True, length=1.5, pad=1)
                axis.tick_params(axis="y", length=1.5, pad=1)
                axis.set_xlabel("Global time index", labelpad=1)
                axis.grid(False)
            figure.subplots_adjust(left=0.12, right=0.995, top=0.995, bottom=0.035, hspace=0.56)
            figure.savefig(
                temporary,
                format="png",
                dpi=dpi,
                facecolor="white",
                metadata={"Software": "Multi-AXIS deterministic renderer"},
            )
        finally:
            plt.close(figure)
    os.replace(temporary, output_path)

    from PIL import Image

    with Image.open(output_path) as rendered:
        rendered.verify()
    with Image.open(output_path) as rendered:
        width, height = map(int, rendered.size)
    return {
        "width": width,
        "height": height,
        "pixels": width * height,
        "sha256": sha256_file(output_path),
        "font_family": font_audit["family"],
        "font_bundle_sha256": font_audit["bundle_sha256"],
    }


def iter_manifest_paths(manifest_dir: Path, requested: Sequence[str]) -> Iterable[Path]:
    if requested:
        for name in requested:
            path = manifest_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            yield path
        return
    for path in sorted(manifest_dir.glob("*.jsonl")):
        if path.name in {"train.jsonl", "validation.jsonl"} or path.name.startswith("eval_"):
            yield path


def render_manifests(
    config: MultiAxisConfig,
    data_root: Path,
    manifest_dir: Path,
    output_root: Path,
    manifest_names: Sequence[str],
) -> Dict[str, Any]:
    if not config.vision.enabled:
        raise ValueError("Image rendering requires vision.enabled=true")
    font_audit = require_times_new_roman()
    output_root.mkdir(parents=True, exist_ok=True)
    existing_records: Dict[str, Dict[str, Any]] = {}
    existing_manifest = output_root / "image_manifest.jsonl"
    progress_manifest = output_root / "image_manifest.progress.jsonl"
    for audit_path in (existing_manifest, progress_manifest):
        if not audit_path.is_file():
            continue
        with audit_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    image_id = str(item["image_id"])
                    if image_id in existing_records and existing_records[image_id] != item:
                        raise RuntimeError(f"Conflicting existing image manifest id: {image_id}")
                    existing_records[image_id] = item
    records: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    manifest_counts: Dict[str, int] = {}
    for manifest_path in iter_manifest_paths(manifest_dir, manifest_names):
        dataset = ManifestDataset(
            manifest_path,
            data_root,
            window_epsilon=config.hints.window_epsilon,
            window_scale=config.hints.window_scale,
            renderer_version=config.vision.renderer_version,
        )
        manifest_counts[manifest_path.name] = len(dataset)
        for index in range(len(dataset)):
            manifest_record = dataset._manifest_record(index)
            interval = tuple(map(int, manifest_record["interval"]))
            image_id = str(
                manifest_record.get("image_id")
                or visual_image_id(
                    str(manifest_record["base_sample_id"]),
                    interval,
                    config.vision.renderer_version,
                )
            )
            identity = {
                "base_sample_id": str(manifest_record["base_sample_id"]),
                "interval": list(interval),
            }
            if image_id in seen:
                if seen[image_id] != identity:
                    raise RuntimeError(f"Image-id collision for {image_id}")
                continue
            seen[image_id] = identity
            sample = dataset[index]
            expected_id = visual_image_id(
                sample["base_sample_id"], sample["interval"], config.vision.renderer_version
            )
            if image_id != expected_id:
                raise RuntimeError("Manifest image id does not match its series/window identity")
            relative = visual_image_relpath(image_id)
            output_path = output_root / relative
            source_sha = normalized_series_sha256(sample["normalized_series"])
            previous = existing_records.get(image_id)
            if output_path.exists():
                if previous is None:
                    raise RuntimeError(
                        f"Existing image has no auditable manifest record: {output_path}"
                    )
                expected_fields = {
                    "normalized_series_sha256": source_sha,
                    "renderer_version": config.vision.renderer_version,
                    "dpi": config.vision.renderer_dpi,
                    "image_relpath": relative,
                    "font_family": font_audit["family"],
                    "font_bundle_sha256": font_audit["bundle_sha256"],
                }
                mismatched = {
                    key: (previous.get(key), value)
                    for key, value in expected_fields.items()
                    if previous.get(key) != value
                }
                actual_sha = sha256_file(output_path)
                if previous.get("sha256") != actual_sha:
                    mismatched["sha256"] = (previous.get("sha256"), actual_sha)
                if mismatched:
                    raise RuntimeError(
                        f"Existing render audit mismatch for {image_id}: {mismatched}. "
                        "Use a fresh output root; this command never overwrites audited images."
                    )
                records.append(previous)
                continue
            render_audit = render_normalized_series(
                sample["normalized_series"],
                sample["interval"],
                sample["channel_ids"],
                output_path,
                dpi=config.vision.renderer_dpi,
            )
            records.append(
                {
                    "format": RENDER_FORMAT,
                    "image_id": image_id,
                    "image_relpath": relative,
                    **identity,
                    "time_count": sample["time_count"],
                    "channel_count": sample["channel_count"],
                    "channel_ids": sample["channel_ids"],
                    "normalized_series_sha256": source_sha,
                    "renderer_version": config.vision.renderer_version,
                    "dpi": config.vision.renderer_dpi,
                    "half_open_boundaries": [interval[0] - 0.5, interval[1] - 0.5],
                    "contains_anomaly_score": False,
                    "contains_label_annotation": False,
                    "colors": {
                        "context": CONTEXT_COLOR,
                        "target": TARGET_COLOR,
                        "target_background": TARGET_BACKGROUND,
                    },
                    **render_audit,
                }
            )
            with progress_manifest.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(records[-1], ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                handle.flush()

    manifest_output = output_root / "image_manifest.jsonl"
    temporary = manifest_output.with_name(manifest_output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, manifest_output)
    summary = {
        "format": RENDER_FORMAT,
        "renderer_version": config.vision.renderer_version,
        "dpi": config.vision.renderer_dpi,
        "unique_images": len(records),
        "manifest_rows": manifest_counts,
        "normalization": "consumed ManifestDataset.normalized_series without re-normalization",
        "font": font_audit,
        "image_manifest_sha256": sha256_file(manifest_output),
    }
    (output_root / "image_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


