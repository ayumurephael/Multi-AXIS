from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AnswerOnlyCausalLMOutput:
    loss: torch.Tensor
    supervised_token_count: int
    visual_token_count: int = 0


def answer_only_causal_nll(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    output_embeddings: nn.Module,
) -> AnswerOnlyCausalLMOutput:
    """Compute exact causal NLL only where the next-token label is supervised."""

    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "hidden_states and labels must have shapes [B, L, D] and [B, L]"
        )
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError(
            "hidden_states and labels must share batch/sequence dimensions"
        )
    shifted_labels = labels[:, 1:].contiguous()
    supervised = shifted_labels.ne(-100)
    supervised_token_count = int(supervised.sum().item())
    if supervised_token_count == 0:
        raise ValueError("Answer-only supervision contains no causal target tokens")
    selected_hidden = hidden_states[:, :-1, :][supervised]
    selected_labels = shifted_labels[supervised]
    answer_logits = output_embeddings(selected_hidden)
    loss = F.cross_entropy(answer_logits.float(), selected_labels, reduction="mean")
    return AnswerOnlyCausalLMOutput(
        loss=loss,
        supervised_token_count=supervised_token_count,
    )
