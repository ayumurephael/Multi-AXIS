from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .attention import FlashCrossAttention


def rms_unit(value: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    mean_square = value.float().square().mean(dim=-1, keepdim=True)
    return (value * torch.rsqrt(mean_square + epsilon)).to(value.dtype)


class MultiAxisHintTuner(nn.Module):
    """The complete trainable part of multivariate AXIS."""

    def __init__(
        self,
        vocab_size: int,
        llm_hidden_size: int,
        d_proj: int,
        config,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.llm_hidden_size = llm_hidden_size
        self.d_proj = d_proj
        self.config = config
        self.use_anomaly_evidence = True
        self.use_joint_hint = True
        self.prototype_mapping = nn.Parameter(
            torch.empty(config.num_prototypes, vocab_size)
        )
        self.anomaly_direction = nn.Parameter(torch.zeros(d_proj))
        self.step_projection = nn.Linear(d_proj, llm_hidden_size)
        self.prototype_attention = FlashCrossAttention(
            llm_hidden_size,
            config.prototype_heads,
            require_flash=config.require_flash_attention,
        )
        self.channel_query = nn.Parameter(torch.empty(1, 1, d_proj))
        self.channel_pool = FlashCrossAttention(
            d_proj,
            config.channel_heads,
            require_flash=config.require_flash_attention,
        )
        self.joint_query = nn.Parameter(torch.empty(1, 1, d_proj))
        self.joint_pool = FlashCrossAttention(
            d_proj,
            config.joint_heads,
            require_flash=config.require_flash_attention,
        )
        self.question_projection = nn.Linear(llm_hidden_size, d_proj, bias=False)
        self.fixed_queries = nn.Parameter(
            torch.empty(1, config.fixed_tokens, llm_hidden_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.prototype_mapping)
        nn.init.zeros_(self.anomaly_direction)
        nn.init.xavier_uniform_(self.step_projection.weight)
        nn.init.zeros_(self.step_projection.bias)
        nn.init.normal_(self.channel_query, mean=0.0, std=0.02)
        nn.init.normal_(self.joint_query, mean=0.0, std=0.02)
        nn.init.zeros_(self.question_projection.weight)
        nn.init.normal_(self.fixed_queries, mean=0.0, std=0.02)

    def build_prototype_bank(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        if word_embeddings.shape != (self.vocab_size, self.llm_hidden_size):
            raise ValueError(
                f"Vocabulary embedding mismatch: {tuple(word_embeddings.shape)} != "
                f"{(self.vocab_size, self.llm_hidden_size)}"
            )
        return torch.matmul(self.prototype_mapping, word_embeddings)

    def forward(
        self,
        local_embeddings: torch.Tensor,
        anomaly_logits: torch.Tensor,
        intervals: Sequence[Tuple[int, int]],
        channel_counts: Sequence[int],
        question_semantics: Optional[torch.Tensor],
        word_embeddings: torch.Tensor,
        prototype_override: Optional[torch.Tensor] = None,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        batch, total_steps, max_channels, d_proj = local_embeddings.shape
        if d_proj != self.d_proj or anomaly_logits.shape != (
            batch,
            total_steps,
            max_channels,
            2,
        ):
            raise ValueError("TimeRCD embeddings/logits have incompatible shapes")
        if len(intervals) != batch or len(channel_counts) != batch:
            raise ValueError(
                "Interval/channel-count batches do not match TimeRCD output"
            )

        signed_evidence = torch.tanh(
            (anomaly_logits[..., 1] - anomaly_logits[..., 0]) / 2.0
        )
        enhanced_samples: List[torch.Tensor] = []
        enhanced_channels: List[torch.Tensor] = []
        for index, ((start, end), channels) in enumerate(
            zip(intervals, channel_counts)
        ):
            if not (0 <= start < end <= total_steps):
                raise ValueError(
                    f"Invalid interval [{start}, {end}) for T={total_steps}"
                )
            if not (1 <= channels <= max_channels):
                raise ValueError(
                    f"Invalid channel count {channels} for C={max_channels}"
                )
            # [L,C,D] -> [C,L,D] -> [C*L,D], exactly channel-major.
            step = (
                local_embeddings[index, start:end, :channels]
                .transpose(0, 1)
                .reshape(-1, d_proj)
            )
            evidence = (
                signed_evidence[index, start:end, :channels]
                .transpose(0, 1)
                .reshape(-1)
            )
            enhanced = rms_unit(
                rms_unit(step, self.config.representation_epsilon)
                + evidence[:, None] * self.anomaly_direction[None, :],
                self.config.representation_epsilon,
            )
            enhanced_samples.append(enhanced)
            enhanced_channels.append(enhanced.view(channels, end - start, d_proj))

        max_step_hints = max(sample.shape[0] for sample in enhanced_samples)
        enhanced_padded = local_embeddings.new_zeros(batch, max_step_hints, self.d_proj)
        step_mask = torch.zeros(
            batch, max_step_hints, dtype=torch.bool, device=local_embeddings.device
        )
        for index, enhanced in enumerate(enhanced_samples):
            count = enhanced.shape[0]
            enhanced_padded[index, :count] = enhanced
            step_mask[index, :count] = True

        prototype_bank = (
            prototype_override
            if prototype_override is not None
            else self.build_prototype_bank(word_embeddings)
        )
        expected = (self.config.num_prototypes, self.llm_hidden_size)
        if prototype_bank.shape != expected:
            raise ValueError(
                f"Prototype bank shape {tuple(prototype_bank.shape)} != {expected}"
            )
        prototypes = prototype_bank.unsqueeze(0).expand(batch, -1, -1)

        step_queries = self.step_projection(enhanced_padded)
        step_padded = self.prototype_attention(step_queries, prototypes, prototypes)
        step_hints = [step_padded[i, step_mask[i]] for i in range(batch)]

        # Pool each channel independently over its complete target-window
        # timeline with one query shared across all channels and samples.
        total_channels = sum(channel_counts)
        max_window_length = max(memory.shape[1] for memory in enhanced_channels)
        channel_memory = local_embeddings.new_zeros(
            total_channels, max_window_length, self.d_proj
        )
        channel_mask = torch.zeros(
            total_channels,
            max_window_length,
            dtype=torch.bool,
            device=local_embeddings.device,
        )
        channel_slices: List[Tuple[int, int]] = []
        cursor = 0
        for memory in enhanced_channels:
            channels, length, _ = memory.shape
            channel_memory[cursor : cursor + channels, :length] = memory
            channel_mask[cursor : cursor + channels, :length] = True
            channel_slices.append((cursor, cursor + channels))
            cursor += channels
        channel_query = self.channel_query.expand(total_channels, -1, -1)
        # Deliberately no residual from the sample-independent shared query.
        channel_repr = self.channel_pool(
            channel_query,
            channel_memory,
            channel_memory,
            key_padding_mask=channel_mask,
        )
        channel_queries = self.step_projection(
            rms_unit(channel_repr, self.config.representation_epsilon)
        )
        channel_prototypes = prototype_bank.unsqueeze(0).expand(
            total_channels, -1, -1
        )
        channel_pooled = self.prototype_attention(
            channel_queries, channel_prototypes, channel_prototypes
        )
        channel_hints = [
            channel_pooled[start:end, 0] for start, end in channel_slices
        ]

        if question_semantics is None:
            raise ValueError(
                "Question-conditioned Joint-Local pooling requires question semantics"
            )
        expected_question_shape = (batch, self.llm_hidden_size)
        if tuple(question_semantics.shape) != expected_question_shape:
            raise ValueError(
                f"Question semantics shape {tuple(question_semantics.shape)} != "
                f"{expected_question_shape}"
            )
        semantic_delta = self.question_projection(
            rms_unit(question_semantics, self.config.representation_epsilon)
        )
        joint_query = self.joint_query.expand(batch, -1, -1) + semantic_delta[:, None]
        # Deliberately no residual connection from the sample-independent query.
        joint_repr = self.joint_pool(
            joint_query,
            enhanced_padded,
            enhanced_padded,
            key_padding_mask=step_mask,
        )
        joint_queries = self.step_projection(
            rms_unit(joint_repr, self.config.representation_epsilon)
        )
        joint_hint = self.prototype_attention(joint_queries, prototypes, prototypes)

        fixed_queries = self.fixed_queries.expand(batch, -1, -1)
        fixed_hints = self.prototype_attention(fixed_queries, prototypes, prototypes)
        return step_hints, channel_hints, joint_hint, fixed_hints
