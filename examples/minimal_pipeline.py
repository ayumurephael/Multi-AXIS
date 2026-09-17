"""CPU structural smoke example for TimeRCD-to-hint evidence construction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multi_axis.config import HintConfig, TimeRCDConfig  # noqa: E402
from multi_axis.hint_tuner import MultiAxisHintTuner  # noqa: E402
from multi_axis.timercd import FrozenTimeRCD  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    timercd = FrozenTimeRCD(
        TimeRCDConfig(
            d_model=32,
            d_proj=16,
            patch_size=4,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            checkpoint_num_features=3,
        )
    )
    series = torch.randn(1, 12, 3)
    time_mask = torch.ones(1, 12, dtype=torch.bool)
    channel_mask = torch.ones(1, 3, dtype=torch.bool)
    local, anomaly_logits = timercd(series, time_mask, channel_mask)

    hint_config = HintConfig(
        num_prototypes=8,
        prototype_heads=4,
        fixed_tokens=2,
        channel_heads=4,
        joint_heads=4,
        require_flash_attention=False,
    )
    tuner = MultiAxisHintTuner(
        vocab_size=32,
        llm_hidden_size=16,
        d_proj=16,
        config=hint_config,
    )
    step, channel, joint, fixed = tuner(
        local,
        anomaly_logits,
        intervals=[(2, 8)],
        channel_counts=[3],
        question_semantics=torch.randn(1, 16),
        word_embeddings=torch.randn(32, 16),
    )
    print(
        {
            "local": tuple(local.shape),
            "anomaly_logits": tuple(anomaly_logits.shape),
            "step_hints": tuple(step[0].shape),
            "channel_hints": tuple(channel[0].shape),
            "joint_hint": tuple(joint.shape),
            "fixed_hints": tuple(fixed.shape),
        }
    )


if __name__ == "__main__":
    main()

