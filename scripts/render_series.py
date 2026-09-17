from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multi_axis.config import MultiAxisConfig  # noqa: E402
from multi_axis.rendering import render_manifests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render deterministic, normalized Multi-AXIS sequence figures."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifests", nargs="*", default=[])
    args = parser.parse_args()
    summary = render_manifests(
        MultiAxisConfig.load_json(args.config),
        args.data_root.resolve(),
        args.manifest_dir.resolve(),
        args.output_root.resolve(),
        args.manifests,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
