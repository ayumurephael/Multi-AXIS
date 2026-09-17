from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multi_axis.data import (  # noqa: E402
    extract_interval,
    extract_values,
    question_type_group,
    teacher_answer,
    visual_image_id,
    visual_image_relpath,
)


def iter_jsonl(path: Path) -> Iterable[tuple[int, int, int, Dict[str, Any]]]:
    with path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            if raw.strip():
                yield line_number, offset, len(raw), json.loads(raw)


def build_record(
    row: Dict[str, Any],
    *,
    source: Path,
    data_root: Path,
    line: int,
    offset: int,
    length: int,
    require_answer: bool,
) -> Dict[str, Any]:
    values = extract_values(row)
    interval = extract_interval(row, values.shape[0])
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError(f"{source}:{line}: missing question")
    sample_id = str(row.get("sample_id") or f"{source.stem}:{line}")
    base_sample_id = str(row.get("base_sample_id") or sample_id)
    answer = teacher_answer(row)
    if require_answer and not answer:
        raise ValueError(f"{source}:{line}: missing teacher/model answer")
    image_id = visual_image_id(base_sample_id, interval)
    record: Dict[str, Any] = {
        "question_path": source.relative_to(data_root).as_posix(),
        "question_offset": offset,
        "question_length": length,
        "question_line": line,
        "sample_id": sample_id,
        "base_sample_id": base_sample_id,
        "question": question,
        "question_group": question_type_group(row),
        "interval": list(interval),
        "teacher_answer": answer or None,
        "image_id": image_id,
        "image_relpath": visual_image_relpath(image_id),
    }
    if "label_reference" in row:
        record["label_reference"] = row["label_reference"]
    return record


def write_manifest(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    groups: dict[str, list[int]] = collections.defaultdict(list)
    with path.open("wb") as handle:
        for index, record in enumerate(records):
            offsets.append(handle.tell())
            groups[record["base_sample_id"]].append(index)
            encoded = json.dumps(
                record, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            handle.write(encoded + b"\n")
    path.with_suffix(path.suffix + ".idx.json").write_text(
        json.dumps(offsets, separators=(",", ":")), encoding="utf-8"
    )
    path.with_suffix(path.suffix + ".groups.json").write_text(
        json.dumps(list(groups.values()), separators=(",", ":")), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Multi-AXIS question JSONL files into a random-access manifest."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-answer", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    records: list[Dict[str, Any]] = []
    for raw_source in args.inputs:
        source = raw_source.resolve()
        source.relative_to(data_root)
        for line, offset, length, row in iter_jsonl(source):
            records.append(
                build_record(
                    row,
                    source=source,
                    data_root=data_root,
                    line=line,
                    offset=offset,
                    length=length,
                    require_answer=args.require_answer,
                )
            )
    if not records:
        raise ValueError("No non-empty JSONL records were found")
    write_manifest(records, args.output.resolve())
    summary = {
        "records": len(records),
        "base_samples": len({record["base_sample_id"] for record in records}),
        "question_groups": dict(
            collections.Counter(record["question_group"] for record in records)
        ),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
