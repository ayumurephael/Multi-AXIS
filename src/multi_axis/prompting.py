from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .channel_ids import (
    canonical_channel_ids,
    canonicalize_question_channel_references,
)
from .response_contracts import OUTPUT_CONTRACTS


STEP_TOKEN = "<STEP_HINT>"
CHANNEL_TOKEN = "<CHANNEL_HINT>"
JOINT_TOKEN = "<JOINT_HINT>"
FIXED_TOKEN = "<FIXED_HINT>"
HINT_TOKENS = (FIXED_TOKEN, STEP_TOKEN, CHANNEL_TOKEN, JOINT_TOKEN)

MULTIMODAL_SYSTEM_TEXT = """You are an expert in multivariate time-series analysis. Answer the exact
question using the visual, numeric, and learned contextual evidence supplied
in the user message.

The figure shows the complete per-channel-normalized sequence. The numeric
Window is a compact, approximately reversible target-interval representation
of the same observations. They are two views of the same evidence, not
independent confirmations.

Use the figure to compare temporal shape, surrounding context, relative
timing, and co-movement across channels. Use the numeric Window for explicit
channel-aligned values and exact time indices. When a value visually estimated
from the figure differs from a listed Window value, use the listed value for
the numeric claim.

Treat learned contextual hints as supporting model evidence, not as
ground-truth labels or calibrated anomaly probabilities. Treat task-prior
hints as general task information rather than sample-specific facts.

Return only the visible final answer and its evidence-based explanation.
Do not expose private chain-of-thought, scratch work, hidden-reasoning tags,
placeholder tokens, image paths, or internal representations. Follow the
Output Contract in the user message exactly."""

TEXT_SYSTEM_TEXT = """You are an expert in multivariate time-series analysis. Answer the exact
question using the numeric and learned contextual evidence supplied in the
user message.

The numeric Window is a compact, approximately reversible representation of
observed values from the target interval.

Treat learned contextual hints as supporting model evidence, not as
ground-truth labels or calibrated anomaly probabilities. Treat task-prior
hints as general task information rather than sample-specific facts.

Return only the visible final answer and its evidence-based explanation.
Do not expose private chain-of-thought, scratch work, hidden-reasoning tags,
placeholder tokens, or internal representations. Follow the Output Contract
in the user message exactly."""

USER_PREAMBLE = """You are solving a multivariate time-series question.

Return only the visible final response required by the Output Contract.
Do not expose private chain-of-thought, scratch work, or hidden-reasoning tags."""

MULTIMODAL_PREFIX = """### Question
{question}

### Visual evidence
The following figure shows the complete per-channel-normalized multivariate
sequence. The highlighted segment is the target interval whose explicit
numeric values and learned contextual hints are provided after the figure.
Inspect it specifically for evidence needed to answer the question above."""

FIGURE_NOTE = """### Figure note
Each row represents one channel, and every row label uses the exact ch_<index>
identifier used by the question and numeric Window. Row order is identical to
the channel order in the numeric Window. The horizontal axis is
the global time index. The black curve shows surrounding context. The dark-blue
segment and pale-yellow background mark exactly the half-open target interval
[{start}, {end}), that is, positions {start} through {last}.

Values are normalized independently within each channel. Plot colors indicate
location and rendering roles only; they do not indicate anomaly status,
severity, affected scope, root cause, or causal direction."""

EVIDENCE_RULES = """### Evidence-use rules
For every channel, mean and std are computed from the full valid series. Each
listed integer is approximately 100 * (raw_value - mean) / (std + epsilon),
where epsilon is stated below. Use normalized integers to assess deviations
relative to each channel's own historical scale.

When the question requires raw levels, absolute changes, or cross-channel
magnitude comparisons, interpret the integers together with that channel's
mean, std, and epsilon using the reconstruction relation stated below.

Do not compare normalized integer magnitudes across different channels as
if they were raw-scale amplitudes. Use normalized integers for deviations
relative to the channel's own history. Use mean and std for raw-scale,
absolute-amplitude, or cross-channel magnitude comparisons.

Each numeric value is immediately followed by one Step-Local hint for the same
channel and the same time step.

Step-Local hints provide fine-grained contextual evidence.
Each Channel-Local hint is a question-independent, channel-anchored summary
of the complete target window. It may reflect temporal context and information
from other channels. The Joint-Local hint is a question-conditioned summary
derived from all channels and all target-window steps.
Fixed hints contain task-level priors, not sample-specific facts.

Use all channels jointly when making the overall judgment.
Use Channel-Local hints for channel-level behavior and comparisons.
Use Window values and Step-Local hints for precise numeric, channel-aligned,
and time-local evidence.
Use the question-conditioned Joint-Local hint for the overall multivariate
interpretation relevant to the current question.
Use Fixed hints only as task-level guidance.
Do not treat the largest deviation as automatically being the root cause.
Do not infer a causal direction solely from anomaly magnitude."""


@dataclass
class TokenizedPrompts:
    model_inputs: Dict[str, torch.Tensor]
    labels: Optional[torch.Tensor]
    prompt_lengths: List[int]
    step_positions: List[torch.Tensor]
    channel_positions: List[torch.Tensor]
    joint_positions: List[torch.Tensor]
    fixed_positions: List[torch.Tensor]
    visual_token_counts: List[int]
    original_image_sizes: List[Optional[Tuple[int, int]]]
    padding_side: str

    @property
    def input_ids(self) -> torch.Tensor:
        return self.model_inputs["input_ids"]

    @property
    def attention_mask(self) -> torch.Tensor:
        return self.model_inputs["attention_mask"]


class MultiAxisPromptBuilder:
    """Build native Qwen3-VL chat inputs without truncating numeric evidence."""

    def __init__(
        self,
        processor,
        fixed_tokens: int = 30,
        max_context_tokens: int = 131072,
        include_joint: bool = True,
        use_images: bool = True,
        min_pixels: int = 65_536,
        max_pixels: int = 2_097_152,
        window_epsilon: float = 1e-5,
    ):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.fixed_tokens = int(fixed_tokens)
        self.include_joint = bool(include_joint)
        self.use_images = bool(use_images)
        self.max_context_tokens = int(max_context_tokens)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.window_epsilon = float(window_epsilon)
        if self.window_epsilon <= 0:
            raise ValueError("window_epsilon must be positive")
        if self.use_images and not 0 < self.min_pixels <= self.max_pixels:
            raise ValueError("Invalid Qwen3-VL runtime pixel budget")
        self.token_ids: Dict[str, int] = {}
        for token in HINT_TOKENS:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{token} must tokenize to exactly one token, got {ids}")
            self.token_ids[token] = int(ids[0])
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token before prompt construction")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")

    @contextmanager
    def _padding_side(self, side: str):
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported padding side: {side}")
        previous = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = side
        try:
            yield
        finally:
            self.tokenizer.padding_side = previous

    def build_text(
        self,
        question: str,
        start: int,
        end: int,
        window_values: Sequence[Sequence[int]],
        channel_means: Sequence[float],
        channel_stds: Sequence[float],
        channel_ids: Optional[Sequence[str]] = None,
        question_group: str = "OE",
        include_visual: Optional[bool] = None,
    ) -> str:
        if not question or not question.strip():
            raise ValueError("Question is empty")
        question = canonicalize_question_channel_references(question)
        length = int(end) - int(start)
        if length <= 0:
            raise ValueError(f"Invalid target interval [{start}, {end})")
        if not window_values:
            raise ValueError("At least one channel is required")
        if any(len(channel) != length for channel in window_values):
            raise ValueError("Every channel must contain exactly end-start target-window values")
        identifiers = canonical_channel_ids(channel_ids, len(window_values))
        if len(channel_means) != len(window_values) or len(channel_stds) != len(
            window_values
        ):
            raise ValueError("Channel statistic count does not match the numeric Window")
        if question_group not in OUTPUT_CONTRACTS:
            raise ValueError(f"Unknown question group: {question_group}")

        visual = self.use_images if include_visual is None else bool(include_visual)
        if visual:
            prefix, suffix = self.build_multimodal_parts(
                question,
                start,
                end,
                window_values,
                channel_means,
                channel_stds,
                identifiers,
                question_group,
            )
            return f"{prefix}\n\n{suffix}"

        fixed, overview_lines, channel_blocks = self._evidence_fields(
            start, window_values, channel_means, channel_stds, identifiers
        )
        sections = [
            USER_PREAMBLE,
            f"### Question\n{question.strip()}",
            EVIDENCE_RULES,
            f"### Task-prior hints\n{fixed}",
            (
                "### All-channel overview\n"
                "Each channel appears once at overview resolution. Channel-Local "
                "hints are question-independent summaries of the complete "
                "target window.\n"
                + "\n".join(overview_lines)
            ),
            (
                "### Target window\n"
                f"Global interval: [{start}, {end})\n"
                f"Window length: {length}\n"
                f"Normalization epsilon: {self._format_stat(self.window_epsilon)}\n"
                "Each integer is rounded from 100 * (raw_value - mean) / "
                "(std + epsilon). Therefore raw_value is approximately mean + "
                "(std + epsilon) * integer / 100."
            ),
            "### Channel-aligned Window and Step-Local evidence\n"
            + "\n\n".join(channel_blocks),
        ]
        if self.include_joint:
            sections.append(
                f"### Question-conditioned full-window Joint-Local evidence\n{JOINT_TOKEN}"
            )
        sections.append(OUTPUT_CONTRACTS[question_group])
        return "\n\n".join(sections)

    def _evidence_fields(
        self,
        start: int,
        window_values: Sequence[Sequence[int]],
        channel_means: Sequence[float],
        channel_stds: Sequence[float],
        identifiers: Sequence[str],
    ) -> Tuple[str, List[str], List[str]]:
        fixed = " ".join([FIXED_TOKEN] * self.fixed_tokens)
        identifiers = canonical_channel_ids(identifiers, len(window_values))
        overview_lines = [f"{identifier}: {CHANNEL_TOKEN}" for identifier in identifiers]
        channel_blocks = []
        for identifier, mean, std, values in zip(
            identifiers, channel_means, channel_stds, window_values
        ):
            lines = [
                f"{identifier} [mean={self._format_stat(mean)}, "
                f"std={self._format_stat(std)}]:"
            ]
            lines.extend(
                f"t={start + relative}, value={int(value)} {STEP_TOKEN}"
                for relative, value in enumerate(values)
            )
            channel_blocks.append("\n".join(lines))
        return fixed, overview_lines, channel_blocks

    @staticmethod
    def _format_stat(value: float) -> str:
        number = float(value)
        if not torch.isfinite(torch.tensor(number)):
            raise ValueError("Channel statistics must be finite")
        return format(number, ".10g")

    def build_multimodal_parts(
        self,
        question: str,
        start: int,
        end: int,
        window_values: Sequence[Sequence[int]],
        channel_means: Sequence[float],
        channel_stds: Sequence[float],
        channel_ids: Sequence[str],
        question_group: str,
    ) -> Tuple[str, str]:
        # Reuse build_text's fail-closed validation without recursively selecting
        # the visual path.
        if not question or not question.strip():
            raise ValueError("Question is empty")
        question = canonicalize_question_channel_references(question)
        length = int(end) - int(start)
        if length <= 0:
            raise ValueError(f"Invalid target interval [{start}, {end})")
        if not window_values or any(len(channel) != length for channel in window_values):
            raise ValueError("Every channel must contain exactly end-start target-window values")
        channel_ids = canonical_channel_ids(channel_ids, len(window_values))
        if len(channel_means) != len(window_values) or len(channel_stds) != len(
            window_values
        ):
            raise ValueError("Channel statistic count does not match the numeric Window")
        if question_group not in OUTPUT_CONTRACTS:
            raise ValueError(f"Unknown question group: {question_group}")
        fixed, overview_lines, channel_blocks = self._evidence_fields(
            start, window_values, channel_means, channel_stds, channel_ids
        )
        prefix = MULTIMODAL_PREFIX.format(question=question.strip())
        suffix_sections = [
            FIGURE_NOTE.format(start=start, end=end, last=end - 1),
            EVIDENCE_RULES,
            f"### Task-prior hints\n{fixed}",
            (
                "### All-channel overview\n"
                "Each channel appears once at overview resolution. Channel-Local "
                "hints are question-independent summaries of the complete "
                "target window.\n"
                + "\n".join(overview_lines)
            ),
            (
                "### Target window\n"
                f"Global interval: [{start}, {end})\n"
                f"Window length: {length}\n"
                f"Normalization epsilon: {self._format_stat(self.window_epsilon)}\n"
                "Each integer is rounded from 100 * (raw_value - mean) / "
                "(std + epsilon). Therefore raw_value is approximately mean + "
                "(std + epsilon) * integer / 100."
            ),
            "### Channel-aligned Window and Step-Local evidence\n"
            + "\n\n".join(channel_blocks),
        ]
        if self.include_joint:
            suffix_sections.append(
                f"### Question-conditioned full-window Joint-Local evidence\n{JOINT_TOKEN}"
            )
        # OUTPUT_CONTRACTS includes its heading, so the suffix must not add one.
        suffix_sections.append(OUTPUT_CONTRACTS[question_group])
        return prefix, "\n\n".join(suffix_sections)

    def _messages(
        self,
        user_text: str,
        image: Any = None,
        *,
        visual_prefix: Optional[str] = None,
        visual_suffix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if image is None:
            # Text-only tokenizers such as DeepSeek/Qwen3 use chat templates that
            # concatenate message content directly and therefore require strings.
            user_content: Any = user_text
        else:
            if visual_prefix is None or visual_suffix is None:
                raise ValueError("Image messages require both text prefix and suffix")
            user_content = [
                {"type": "text", "text": visual_prefix},
                {"type": "image", "image": image},
                {"type": "text", "text": visual_suffix},
            ]
        return [
            {
                "role": "system",
                "content": MULTIMODAL_SYSTEM_TEXT if image is not None else TEXT_SYSTEM_TEXT,
            },
            {"role": "user", "content": user_content},
        ]

    def _render_chat(self, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
        renderer = getattr(self.processor, "apply_chat_template", None)
        if renderer is None:
            renderer = getattr(self.tokenizer, "apply_chat_template", None)
        if renderer is None:
            raise TypeError("Formal prompt construction requires a native chat template")
        template = getattr(self.processor, "chat_template", None)
        if template is None:
            template = getattr(self.tokenizer, "chat_template", "")
        if isinstance(template, dict):
            template = "\n".join(map(str, template.values()))
        kwargs = {"enable_thinking": False} if "enable_thinking" in str(template) else {}
        return str(
            renderer(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
        )

    def forbidden_reasoning_token_sequences(self) -> List[List[int]]:
        sequences: List[List[int]] = []
        for marker in ("<think>", "</think>"):
            ids = list(self.tokenizer.encode(marker, add_special_tokens=False))
            if ids and ids not in sequences:
                sequences.append(ids)
        return sequences

    def _open_images(
        self, image_paths: Sequence[Optional[str]]
    ) -> Tuple[List[Any], List[Optional[Tuple[int, int]]]]:
        if not self.use_images:
            if any(path is not None for path in image_paths):
                raise ValueError("Image-off control received image paths")
            return [], [None] * len(image_paths)
        if any(not path for path in image_paths):
            raise FileNotFoundError("Every image-enabled sample requires a pre-rendered image")
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for Qwen3-VL image inputs") from exc
        images: List[Any] = []
        sizes: List[Optional[Tuple[int, int]]] = []
        for raw_path in image_paths:
            path = Path(str(raw_path))
            if not path.is_file():
                raise FileNotFoundError(f"Pre-rendered VLM image does not exist: {path}")
            with Image.open(path) as source:
                image = source.convert("RGB").copy()
            images.append(image)
            sizes.append(tuple(map(int, image.size)))
        return images, sizes

    def _processor_call(
        self, texts: Sequence[str], images: Sequence[Any], padding: bool
    ) -> Dict[str, torch.Tensor]:
        if not callable(self.processor):
            raise TypeError("Formal VLM prompt construction requires a callable processor")
        kwargs: Dict[str, Any] = {
            "text": list(texts),
            "padding": padding,
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = list(images)
            kwargs["return_mm_token_type_ids"] = True
            # Pass the budget on every call. Transformers 4.57 may otherwise
            # recover the AutoProcessor construction-time value from tokenizer
            # init kwargs instead of the current image-processor size.
            kwargs["min_pixels"] = self.min_pixels
            kwargs["max_pixels"] = self.max_pixels
        features = self.processor(**kwargs)
        native_keys = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        }
        tensors = {
            key: value
            for key, value in dict(features).items()
            if key in native_keys and torch.is_tensor(value)
        }
        if "input_ids" not in tensors or "attention_mask" not in tensors:
            raise RuntimeError("Processor omitted input_ids or attention_mask")
        if images:
            required = {"pixel_values", "image_grid_thw", "mm_token_type_ids"}
            missing = sorted(required - set(tensors))
            if missing:
                raise RuntimeError(f"Qwen3-VL processor omitted required tensors: {missing}")
        return tensors

    def _prompt_ids(
        self,
        prompt_text: str,
        image_grid_thw: Optional[torch.Tensor],
        image: Any,
    ) -> List[int]:
        expanded = prompt_text
        if image_grid_thw is not None:
            replace = getattr(self.processor, "replace_image_token", None)
            if callable(replace):
                replacement = replace({"image_grid_thw": image_grid_thw}, 0)
                image_token = str(getattr(self.processor, "image_token", "<|image_pad|>"))
                if image_token not in prompt_text:
                    raise RuntimeError("Native chat template omitted the Qwen3-VL image token")
                expanded = prompt_text.replace(image_token, replacement, 1)
            else:
                single = self._processor_call([prompt_text], [image], padding=False)
                return single["input_ids"][0].tolist()
        return list(self.tokenizer.encode(expanded, add_special_tokens=False))

    def tokenize(
        self,
        questions: Sequence[str],
        intervals: Sequence[Tuple[int, int]],
        window_values: Sequence[Sequence[Sequence[int]]],
        channel_means: Sequence[Sequence[float]],
        channel_stds: Sequence[Sequence[float]],
        channel_ids: Sequence[Sequence[str]],
        question_groups: Sequence[str],
        image_paths: Sequence[Optional[str]],
        answers: Optional[Sequence[str]] = None,
    ) -> TokenizedPrompts:
        batch = len(questions)
        related = (
            intervals,
            window_values,
            channel_means,
            channel_stds,
            channel_ids,
            question_groups,
            image_paths,
        )
        if any(len(values) != batch for values in related):
            raise ValueError("Prompt batch fields must have identical lengths")
        if answers is not None and len(answers) != batch:
            raise ValueError("Answer batch length must match questions")
        images, original_sizes = self._open_images(image_paths)

        prompt_texts: List[str] = []
        full_texts: List[str] = []
        for index, question in enumerate(questions):
            start, end = intervals[index]
            image = images[index] if images else None
            if image is None:
                user_text = self.build_text(
                    question,
                    start,
                    end,
                    window_values[index],
                    channel_means[index],
                    channel_stds[index],
                    channel_ids[index],
                    question_groups[index],
                    include_visual=False,
                )
                messages = self._messages(user_text)
            else:
                visual_prefix, visual_suffix = self.build_multimodal_parts(
                    question,
                    start,
                    end,
                    window_values[index],
                    channel_means[index],
                    channel_stds[index],
                    channel_ids[index],
                    question_groups[index],
                )
                messages = self._messages(
                    "",
                    image,
                    visual_prefix=visual_prefix,
                    visual_suffix=visual_suffix,
                )
            prompt_text = self._render_chat(messages, add_generation_prompt=True)
            if image_paths[index] and str(image_paths[index]) in prompt_text:
                raise RuntimeError("A local image path leaked into the textual prompt")
            prompt_texts.append(prompt_text)
            if answers is None:
                full_texts.append(prompt_text)
            else:
                answer = answers[index]
                if not answer or not answer.strip():
                    raise ValueError("Empty teacher answer reached model forward")
                assistant_content: Any = (
                    [{"type": "text", "text": answer.strip()}]
                    if self.use_images
                    else answer.strip()
                )
                full_messages = messages + [
                    {"role": "assistant", "content": assistant_content}
                ]
                full_texts.append(
                    self._render_chat(full_messages, add_generation_prompt=False)
                )

        padding_side = "right" if answers is not None else "left"
        with self._padding_side(padding_side):
            model_inputs = self._processor_call(full_texts, images, padding=True)

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = torch.full_like(input_ids, -100) if answers is not None else None
        prompt_lengths: List[int] = []
        grids = model_inputs.get("image_grid_thw")
        if images and (grids is None or grids.shape[0] != batch):
            raise RuntimeError("Exactly one image grid is required per VLM sample")

        step_positions: List[torch.Tensor] = []
        channel_positions: List[torch.Tensor] = []
        joint_positions: List[torch.Tensor] = []
        fixed_positions: List[torch.Tensor] = []
        visual_token_counts: List[int] = []
        merge_size = int(getattr(getattr(self.processor, "image_processor", None), "merge_size", 2))
        for index in range(batch):
            valid_length = int(attention_mask[index].sum().item())
            if valid_length > self.max_context_tokens:
                raise ValueError(
                    f"Prompt length {valid_length} exceeds {self.max_context_tokens}; "
                    "Multi-AXIS never truncates prompts."
                )
            row_start = 0 if padding_side == "right" else input_ids.shape[1] - valid_length
            full_ids = input_ids[index, row_start : row_start + valid_length].tolist()
            grid = grids[index : index + 1] if grids is not None else None
            prompt_ids = self._prompt_ids(
                prompt_texts[index], grid, images[index] if images else None
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(
                    "Native chat-template prompt is not an exact prefix of the supervised sequence"
                )
            prompt_lengths.append(len(prompt_ids))
            if labels is not None:
                labels[index, row_start : row_start + valid_length] = input_ids[
                    index, row_start : row_start + valid_length
                ]
                labels[index, row_start : row_start + len(prompt_ids)] = -100

            row_ids = input_ids[index]
            valid = attention_mask[index].bool()
            step = torch.where((row_ids == self.token_ids[STEP_TOKEN]) & valid)[0]
            channel = torch.where(
                (row_ids == self.token_ids[CHANNEL_TOKEN]) & valid
            )[0]
            joint = torch.where((row_ids == self.token_ids[JOINT_TOKEN]) & valid)[0]
            fixed = torch.where((row_ids == self.token_ids[FIXED_TOKEN]) & valid)[0]
            expected_steps = len(window_values[index]) * (intervals[index][1] - intervals[index][0])
            if step.numel() != expected_steps:
                raise AssertionError(f"STEP placeholder mismatch: {step.numel()} != {expected_steps}")
            expected_channels = len(window_values[index])
            if channel.numel() != expected_channels:
                raise AssertionError(
                    f"CHANNEL placeholder mismatch: {channel.numel()} != {expected_channels}"
                )
            expected_joint = 1 if self.include_joint else 0
            if joint.numel() != expected_joint:
                raise AssertionError(f"JOINT placeholder mismatch: {joint.numel()} != {expected_joint}")
            if fixed.numel() != self.fixed_tokens:
                raise AssertionError(
                    f"FIXED placeholder mismatch: {fixed.numel()} != {self.fixed_tokens}"
                )
            step_positions.append(step)
            channel_positions.append(channel)
            joint_positions.append(joint)
            fixed_positions.append(fixed)
            visual_token_counts.append(
                int(grid.prod().item() // (merge_size**2)) if grid is not None else 0
            )

        if labels is not None and int(labels.ne(-100).sum().item()) == 0:
            raise ValueError("No answer tokens remain after chat-template masking")
        return TokenizedPrompts(
            model_inputs=model_inputs,
            labels=labels,
            prompt_lengths=prompt_lengths,
            step_positions=step_positions,
            channel_positions=channel_positions,
            joint_positions=joint_positions,
            fixed_positions=fixed_positions,
            visual_token_counts=visual_token_counts,
            original_image_sizes=original_sizes,
            padding_side=padding_side,
        )
