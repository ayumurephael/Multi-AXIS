from __future__ import annotations

import numpy as np
import torch

from multi_axis.channel_ids import (
    canonical_channel_ids,
    canonicalize_question_channel_references,
)
from multi_axis.config import HintConfig, TimeRCDConfig
from multi_axis.data import normalize_and_serialize
from multi_axis.hint_tuner import MultiAxisHintTuner
from multi_axis.response_contracts import parse_response_label, response_contract_error
from multi_axis.timercd import FrozenTimeRCD


def test_normalization_and_channel_protocol() -> None:
    values = np.asarray([[1.0, 10.0], [2.0, 12.0], [3.0, 14.0]], dtype=np.float32)
    normalized, window, means, stds = normalize_and_serialize(values, (1, 3))
    assert normalized.shape == (3, 2)
    assert len(window) == 2 and all(len(channel) == 2 for channel in window)
    assert means == [2.0, 12.0]
    assert len(stds) == 2
    assert canonical_channel_ids(["Channel-0", "ch 1"], 2) == ["ch_0", "ch_1"]
    assert canonicalize_question_channel_references("Compare channel 0 and CH-1") == (
        "Compare ch_0 and ch_1"
    )


def test_response_contracts() -> None:
    response = "Answer: B\n\nThe second option follows from the stated evidence."
    assert parse_response_label(response, "MC") == "B"
    assert response_contract_error(response, "MC") is None
    assert response_contract_error("Preamble\nAnswer: B", "MC") == "mc_first_line"


def test_timercd_and_full_hint_path_shapes() -> None:
    torch.manual_seed(3)
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
    local, logits = timercd(
        torch.randn(1, 12, 3),
        torch.ones(1, 12, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert local.shape == (1, 12, 3, 16)
    assert logits.shape == (1, 12, 3, 2)

    tuner = MultiAxisHintTuner(
        vocab_size=32,
        llm_hidden_size=16,
        d_proj=16,
        config=HintConfig(
            num_prototypes=8,
            prototype_heads=4,
            fixed_tokens=2,
            channel_heads=4,
            joint_heads=4,
            require_flash_attention=False,
        ),
    )
    step, channel, joint, fixed = tuner(
        local,
        logits,
        intervals=[(2, 8)],
        channel_counts=[3],
        question_semantics=torch.randn(1, 16),
        word_embeddings=torch.randn(32, 16),
    )
    assert step[0].shape == (18, 16)
    assert channel[0].shape == (3, 16)
    assert joint.shape == (1, 1, 16)
    assert fixed.shape == (1, 2, 16)

