from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multi_axis.config import MultiAxisConfig  # noqa: E402
from multi_axis.data import ManifestDataset  # noqa: E402
from multi_axis.inference import predict_dataset  # noqa: E402
from multi_axis.model import MultiAxisForConditionalGeneration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Multi-AXIS inference path.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--timercd-checkpoint", type=Path, required=True)
    parser.add_argument("--hint-checkpoint", type=Path, required=True)
    parser.add_argument("--question-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = MultiAxisConfig.load_json(args.config)
    if config.vision.enabled and args.image_root is None:
        raise ValueError("--image-root is required when vision.enabled=true")
    device = torch.device(args.device)
    dataset = ManifestDataset(
        args.manifest,
        args.data_root,
        window_epsilon=config.hints.window_epsilon,
        window_scale=config.hints.window_scale,
        image_root=args.image_root,
        require_image=config.vision.enabled,
        renderer_version=config.vision.renderer_version,
    )
    model = MultiAxisForConditionalGeneration.from_pretrained(
        config,
        token=os.environ.get("HF_TOKEN") or None,
        device=device,
    )
    model.configure_question_semantic_cache(args.question_cache)
    model.load_timercd_checkpoint(args.timercd_checkpoint)
    model.load_hint_checkpoint(args.hint_checkpoint, strict=True)
    records = predict_dataset(model, dataset, device, output_path=args.output)
    print(
        json.dumps(
            {"records": len(records), "output": str(args.output.resolve())},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
