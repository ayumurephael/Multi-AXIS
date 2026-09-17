from __future__ import annotations

import contextlib
from dataclasses import asdict
import os
from pathlib import Path
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .attention import FlashCrossAttention
from .channel_ids import canonicalize_question_channel_references
from .config import MultiAxisConfig, QWEN3_VL_MODEL_ID
from .prompting import HINT_TOKENS, MultiAxisPromptBuilder, TokenizedPrompts
from .question_semantics import QuestionSemanticDiskCache, question_semantic_text
from .timercd import FrozenTimeRCD


from .hint_tuner import MultiAxisHintTuner
from .losses import answer_only_causal_nll


class MultiAxisForConditionalGeneration(nn.Module):
    def __init__(self, llm: nn.Module, processor, config: MultiAxisConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.llm = llm
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self._prepare_tokenizer()
        self._freeze_llm()
        hidden_size = self._hidden_size()
        vocab_size = self.llm.get_input_embeddings().weight.shape[0]
        self.timercd = FrozenTimeRCD(config.timercd)
        self.hint_tuner = MultiAxisHintTuner(
            vocab_size=vocab_size,
            llm_hidden_size=hidden_size,
            d_proj=config.timercd.d_proj,
            config=config.hints,
        )
        self.prompt_builder = MultiAxisPromptBuilder(
            processor,
            fixed_tokens=config.hints.fixed_tokens,
            max_context_tokens=config.llm.max_context_tokens,
            include_joint=True,
            use_images=config.vision.enabled,
            min_pixels=config.vision.min_pixels,
            max_pixels=config.vision.runtime_max_pixels,
            window_epsilon=config.hints.window_epsilon,
        )
        self.question_semantic_cache: Optional[QuestionSemanticDiskCache] = None
        self.eval_checkpointed_decoder_layers = 0
        if config.vision.enabled and config.llm.gradient_checkpointing:
            self.eval_checkpointed_decoder_layers = (
                self._enable_eval_safe_activation_checkpointing()
            )

    @classmethod
    def from_pretrained(
        cls,
        config: MultiAxisConfig,
        token: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        dtype = getattr(torch, config.llm.torch_dtype)
        source = os.environ.get("MULTI_AXIS_MODEL_PATH") or config.llm.model_name
        if source != config.llm.model_name and not Path(source).is_dir():
            raise FileNotFoundError(
                f"MULTI_AXIS_MODEL_PATH is not a directory: {source}"
            )
        kwargs = dict(
            token=token,
            trust_remote_code=config.llm.trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=config.llm.attention_implementation,
        )
        if device is not None:
            kwargs["device_map"] = {"": str(device)}
        if config.llm.model_name == QWEN3_VL_MODEL_ID:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            processor = AutoProcessor.from_pretrained(
                source,
                token=token,
                trust_remote_code=config.llm.trust_remote_code,
                min_pixels=config.vision.min_pixels,
                max_pixels=config.vision.max_pixels,
            )
            llm = Qwen3VLForConditionalGeneration.from_pretrained(source, **kwargs)
        else:
            # A text-only checkpoint does not require torchvision. Some older
            # cluster Python builds expose a broken system torchvision (for
            # example, a stdlib build without _lzma); letting Transformers probe
            # it prevents Qwen3ForCausalLM from loading even though no image path
            # is used. Mark the optional dependency unavailable before lazy model
            # imports, while keeping it mandatory for the native VLM branch above.
            from transformers.utils import import_utils as transformers_import_utils

            transformers_import_utils._torchvision_available = False
            from transformers import AutoModelForCausalLM, AutoTokenizer

            processor = AutoTokenizer.from_pretrained(
                source,
                token=token,
                trust_remote_code=config.llm.trust_remote_code,
                use_fast=True,
            )
            try:
                llm = AutoModelForCausalLM.from_pretrained(source, **kwargs)
            except ValueError:
                from transformers import AutoModelForImageTextToText

                llm = AutoModelForImageTextToText.from_pretrained(source, **kwargs)
        instance = cls(llm=llm, processor=processor, config=config)
        instance.pretrained_source = str(source)
        return instance

    def _hidden_size(self) -> int:
        if hasattr(self.llm.config, "hidden_size"):
            return int(self.llm.config.hidden_size)
        text_config = getattr(self.llm.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            hidden_size = int(text_config.hidden_size)
            if self.config.llm.model_name == QWEN3_VL_MODEL_ID and hidden_size != 4096:
                raise ValueError(
                    f"Qwen3-VL-8B text hidden size must be 4096, got {hidden_size}"
                )
            return hidden_size
        raise AttributeError("Could not determine the LLM text hidden size")

    def _prepare_tokenizer(self) -> None:
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(HINT_TOKENS)}
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad nor EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        embedding_rows = int(self.llm.get_input_embeddings().weight.shape[0])
        if added and (
            not self.config.vision.enabled or len(self.tokenizer) > embedding_rows
        ):
            self.llm.resize_token_embeddings(len(self.tokenizer))
        for token in HINT_TOKENS:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise AssertionError(
                    f"Registered hint token {token} is not atomic: {ids}"
                )

    def _freeze_llm(self) -> None:
        for parameter in self.llm.parameters():
            parameter.requires_grad_(False)
        self.llm.eval()
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = self.config.llm.use_cache_during_training
        if (
            not self.config.vision.enabled
            and self.config.llm.gradient_checkpointing
            and hasattr(self.llm, "gradient_checkpointing_enable")
        ):
            try:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.llm.gradient_checkpointing_enable()

    def _enable_eval_safe_activation_checkpointing(self) -> int:
        """Checkpoint frozen text layers without switching the VLM out of eval.

        Transformers' built-in checkpointing is conditional on ``module.training``.
        The method specification requires the frozen VLM to remain in eval mode,
        so wrap the language decoder layers explicitly with non-reentrant PyTorch
        checkpointing instead. This preserves autograd back to injected hints.
        """

        model = getattr(self.llm, "model", None)
        language_model = getattr(model, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is None or len(layers) == 0:
            raise RuntimeError(
                "Could not locate Qwen3-VL language decoder layers for activation checkpointing"
            )
        installed = 0
        for layer in layers:
            if getattr(layer, "_multi_axis_eval_checkpointing", False):
                installed += 1
                continue
            original_forward = layer.forward

            def checkpointed_forward(
                module,
                *args,
                _original_forward=original_forward,
                **kwargs,
            ):
                del module
                if torch.is_grad_enabled():
                    return checkpoint(
                        _original_forward,
                        *args,
                        use_reentrant=False,
                        **kwargs,
                    )
                return _original_forward(*args, **kwargs)

            layer.forward = types.MethodType(checkpointed_forward, layer)
            layer._multi_axis_eval_checkpointing = True
            installed += 1
        return installed

    def _set_llm_runtime_mode(self, training: bool) -> None:
        if self.config.vision.enabled:
            # The VLM specification requires the entire frozen native VLM to stay
            # in eval mode. Autograd still reaches the injected soft hints.
            self.llm.eval()
        elif training and self.config.llm.gradient_checkpointing:
            # Preserve the text-only Multi-AXIS behavior: Transformers activates
            # decoder checkpointing only in train mode, while frozen dropout stays
            # disabled for deterministic hint tuning.
            self.llm.train(True)
            for module in self.llm.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
        else:
            self.llm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_llm_runtime_mode(mode)
        self.timercd.eval()
        self.hint_tuner.train(mode)
        return self

    def _llm_backbone(self) -> nn.Module:
        backbone = getattr(self.llm, "model", None)
        if backbone is None:
            prefix = getattr(self.llm, "base_model_prefix", "")
            backbone = getattr(self.llm, prefix, None) if prefix else None
        if backbone is None or backbone is self.llm:
            raise AttributeError(
                "Could not locate the causal LLM backbone for sparse answer NLL"
            )
        return backbone

    def attention_backend_audit(self) -> Dict[str, Any]:
        """Fail closed unless every formal attention path resolves to FlashAttention 2."""
        requested = self.config.llm.attention_implementation
        if not self.config.hints.require_flash_attention:
            raise RuntimeError("Formal execution requires require_flash_attention=true")
        if requested != "flash_attention_2":
            raise RuntimeError(
                f"Formal execution requested attention backend {requested!r}, not flash_attention_2"
            )

        config_objects = [("llm.config", self.llm.config)]
        text_config = getattr(self.llm.config, "text_config", None)
        if text_config is not None and text_config is not self.llm.config:
            config_objects.append(("llm.config.text_config", text_config))
        resolved_configs = {}
        for name, value in config_objects:
            resolved = getattr(value, "_attn_implementation", None)
            internal = getattr(value, "_attn_implementation_internal", None)
            resolved_configs[name] = {
                "resolved": resolved,
                "internal": internal,
            }
        effective = [
            item["resolved"] or item["internal"] for item in resolved_configs.values()
        ]
        if not effective or any(item != "flash_attention_2" for item in effective):
            raise RuntimeError(
                "LLM attention backend did not resolve to flash_attention_2: "
                f"{resolved_configs}"
            )

        attention_modules = []
        for name, module in self.llm.named_modules():
            class_name = type(module).__name__
            if name.endswith("self_attn") or class_name.lower().endswith("attention"):
                module_config = getattr(module, "config", None)
                if module_config is None:
                    continue
                resolved = getattr(module_config, "_attn_implementation", None)
                internal = getattr(module_config, "_attn_implementation_internal", None)
                implementation = resolved or internal
                attention_modules.append(
                    {
                        "name": name,
                        "class": class_name,
                        "implementation": implementation,
                    }
                )
        if not attention_modules:
            raise RuntimeError(
                "Could not identify any LLM attention modules for backend audit"
            )
        invalid_modules = [
            item
            for item in attention_modules
            if item["implementation"] != "flash_attention_2"
        ]
        if invalid_modules:
            raise RuntimeError(
                "One or more LLM attention modules resolved away from flash_attention_2: "
                f"{invalid_modules[:5]}"
            )

        try:
            import flash_attn
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except Exception as error:
            raise RuntimeError("FlashAttention kernels are not importable") from error
        if not callable(flash_attn_func) or not callable(flash_attn_varlen_func):
            raise RuntimeError(
                "FlashAttention dense/varlen kernel entry points are unavailable"
            )

        named_cross_attention_modules = {
            name: module
            for name, module in self.hint_tuner.named_modules()
            if isinstance(module, FlashCrossAttention)
        }
        expected_cross_attention_modules = {
            "prototype_attention",
            "channel_pool",
            "joint_pool",
        }
        if set(named_cross_attention_modules) != expected_cross_attention_modules:
            raise RuntimeError(
                "Hint Tuner cross-attention inventory is incomplete or unexpected: "
                f"{sorted(named_cross_attention_modules)}"
            )
        if any(
            not module.require_flash
            for module in named_cross_attention_modules.values()
        ):
            raise RuntimeError(
                "Hint Tuner cross-attention is not fail-closed on FlashAttention"
            )
        eager_cross_attention = [
            name
            for name, module in self.hint_tuner.named_modules()
            if isinstance(module, nn.MultiheadAttention)
        ]
        if eager_cross_attention:
            raise RuntimeError(
                "Ordinary MultiheadAttention remains in the Hint Tuner: "
                f"{eager_cross_attention}"
            )

        return {
            "requested": requested,
            "llm_resolved_configs": resolved_configs,
            "llm_attention_module_count": len(attention_modules),
            "llm_attention_module_classes": sorted(
                {item["class"] for item in attention_modules}
            ),
            "hint_cross_attention_module_count": len(named_cross_attention_modules),
            "hint_cross_attention_modules": sorted(named_cross_attention_modules),
            "hint_cross_attention_backend": "flash-attn-v2-package",
            "flash_attn_version": getattr(flash_attn, "__version__", "unknown"),
            "dense_kernel_importable": True,
            "varlen_kernel_importable": True,
            "fallback_allowed": False,
        }

    def trainable_parameter_names(self) -> List[str]:
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def assert_freeze_contract(self) -> None:
        leaked = [name for name, p in self.llm.named_parameters() if p.requires_grad]
        leaked += [
            f"timercd.{name}"
            for name, p in self.timercd.named_parameters()
            if p.requires_grad
        ]
        if leaked:
            raise AssertionError(f"Frozen parameter contract violated: {leaked[:10]}")
        expected_prefix = "hint_tuner."
        unexpected = [
            name
            for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith(expected_prefix)
        ]
        if unexpected:
            raise AssertionError(f"Unexpected trainable parameters: {unexpected}")

    def load_timercd_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        return self.timercd.load_checkpoint(checkpoint_path, strict=True)

    def build_prototype_bank(self) -> torch.Tensor:
        return self.hint_tuner.build_prototype_bank(
            self.llm.get_input_embeddings().weight
        )

    def configure_question_semantic_cache(self, cache_dir: str | Path) -> None:
        self.question_semantic_cache = QuestionSemanticDiskCache(
            cache_dir,
            model_id=self.config.llm.model_name,
            tokenizer=self.tokenizer,
            hidden_size=self._hidden_size(),
        )

    def question_semantic_cache_manifest(self) -> Dict[str, object]:
        if self.question_semantic_cache is None:
            return {"configured": False}
        return {"configured": True, **self.question_semantic_cache.manifest()}

    def _question_text_backbone(self) -> nn.Module:
        if self.config.vision.enabled:
            model = getattr(self.llm, "model", None)
            language_model = getattr(model, "language_model", None)
            if language_model is None:
                raise AttributeError(
                    "Could not locate the frozen Qwen3-VL language model for h_q"
                )
            return language_model
        return self._llm_backbone()

    def _compute_question_semantics(
        self, questions: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        texts = [question_semantic_text(question) for question in questions]
        previous_padding = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "right"
        try:
            encoded = self.tokenizer(
                texts,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = previous_padding
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
        last_indices = attention_mask.sum(dim=-1) - 1
        if bool((last_indices < 0).any()):
            raise ValueError("Question tokenization produced an empty sequence")
        self.llm.eval()
        autocast = (
            torch.autocast("cuda", dtype=getattr(torch, self.config.llm.torch_dtype))
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        try:
            with torch.no_grad(), autocast:
                outputs = self._question_text_backbone()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                result = outputs.last_hidden_state[
                    torch.arange(len(texts), device=device), last_indices
                ]
        finally:
            self._set_llm_runtime_mode(self.training)
        return result.detach()

    def _question_semantics(
        self, questions: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        if self.question_semantic_cache is None:
            raise RuntimeError(
                "Question-conditioned Joint-Local requires a configured disk cache"
            )
        values: List[Optional[torch.Tensor]] = [
            self.question_semantic_cache.load(question) for question in questions
        ]
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            computed = self._compute_question_semantics(
                [questions[index] for index in missing], device
            )
            for row, index in enumerate(missing):
                values[index] = self.question_semantic_cache.store(
                    questions[index], computed[row]
                )
        if any(value is None for value in values):
            raise AssertionError("Question semantic cache fill was incomplete")
        stacked = torch.stack([value for value in values if value is not None])
        return stacked.to(device=device, non_blocking=True).detach()

    def _hint_embeddings(
        self,
        normalized_series: torch.Tensor,
        time_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        intervals: Sequence[Tuple[int, int]],
        channel_counts: Sequence[int],
        questions: Sequence[str],
        prototype_override: Optional[torch.Tensor],
        timercd_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        if timercd_override is None:
            local, anomaly_logits = self.timercd(
                normalized_series, time_mask, channel_mask
            )
        else:
            local, anomaly_logits = timercd_override
        question_semantics = self._question_semantics(
            questions, normalized_series.device
        )
        return self.hint_tuner(
            local,
            anomaly_logits,
            intervals,
            channel_counts,
            question_semantics,
            self.llm.get_input_embeddings().weight,
            prototype_override=prototype_override,
        )

    def _model_inputs_and_hint_hook(
        self,
        tokenized: TokenizedPrompts,
        step_hints: Sequence[torch.Tensor],
        channel_hints: Sequence[torch.Tensor],
        joint_hints: Optional[torch.Tensor],
        fixed_hints: torch.Tensor,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Any]:
        model_inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in tokenized.model_inputs.items()
        }
        input_ids = model_inputs["input_ids"]
        labels = tokenized.labels.to(device) if tokenized.labels is not None else None
        if self.config.vision.enabled:
            required = {"pixel_values", "image_grid_thw", "mm_token_type_ids"}
            missing = sorted(required - set(model_inputs))
            if missing:
                raise RuntimeError(f"Native Qwen3-VL inputs are incomplete: {missing}")
        elif any(key in model_inputs for key in ("pixel_values", "image_grid_thw")):
            raise RuntimeError("Image-off control unexpectedly received visual tensors")

        step_positions = [positions.to(device) for positions in tokenized.step_positions]
        channel_positions = [
            positions.to(device) for positions in tokenized.channel_positions
        ]
        joint_positions = [positions.to(device) for positions in tokenized.joint_positions]
        fixed_positions = [positions.to(device) for positions in tokenized.fixed_positions]
        for index in range(input_ids.shape[0]):
            if step_hints[index].shape[0] != step_positions[index].numel():
                raise AssertionError("Step hint count changed between prompt and encoder")
            if channel_hints[index].shape[0] != channel_positions[index].numel():
                raise AssertionError("Channel hint count changed between prompt and encoder")
            if fixed_hints[index].shape[0] != fixed_positions[index].numel():
                raise AssertionError("Fixed hint count changed between prompt and encoder")
            if joint_hints is not None and joint_positions[index].numel() != 1:
                raise AssertionError("Joint hint placeholder is missing")

        def hint_hook(module, args, base_embeddings):
            del module
            token_ids = args[0]
            # Preserve all Qwen3-VL native input ids. Incremental generation and
            # internal special-token lookups must not receive hint replacement.
            # Beam generation repeat-interleaves complete prompts before its first
            # forward, so accept only an exact full-prompt repeat expansion.
            if (
                token_ids.ndim != 2
                or token_ids.shape[1] != input_ids.shape[1]
                or token_ids.shape[0] % input_ids.shape[0] != 0
            ):
                return base_embeddings
            expansion = token_ids.shape[0] // input_ids.shape[0]
            expected_ids = input_ids.repeat_interleave(expansion, dim=0)
            if not torch.equal(token_ids, expected_ids):
                return base_embeddings
            output = base_embeddings.clone()
            for index in range(token_ids.shape[0]):
                source_index = index // expansion
                output[index, fixed_positions[source_index]] = fixed_hints[source_index].to(
                    device=output.device, dtype=output.dtype
                )
                output[index, step_positions[source_index]] = step_hints[source_index].to(
                    device=output.device, dtype=output.dtype
                )
                output[index, channel_positions[source_index]] = channel_hints[
                    source_index
                ].to(device=output.device, dtype=output.dtype)
                if joint_hints is not None:
                    output[index, joint_positions[source_index]] = joint_hints[source_index].to(
                        device=output.device, dtype=output.dtype
                    )
            return output

        return model_inputs, labels, hint_hook

    @contextlib.contextmanager
    def _native_generation_metadata_context(
        self, model_inputs: Dict[str, torch.Tensor]
    ):
        """Bridge processor MM metadata through Transformers generation validation.

        Transformers 4.57 Qwen3-VL accepts ``mm_token_type_ids`` through the
        model's typed kwargs, but its generation prepare signature does not list
        the key. Keep the official tensor for the visual prefill instead of
        deleting it merely to satisfy ``GenerationMixin`` validation.
        """

        if not self.config.vision.enabled:
            audit = {"required": False, "prefill_calls": 0, "metadata_present": True}
            yield audit
            self.last_generation_metadata_audit = dict(audit)
            return
        if "mm_token_type_ids" not in model_inputs:
            raise RuntimeError("Native generation requires mm_token_type_ids")
        original_prepare = self.llm.prepare_inputs_for_generation
        audit = {"required": True, "prefill_calls": 0, "metadata_present": True}

        def prepare_with_mm_token_types(
            module,
            input_ids,
            mm_token_type_ids=None,
            **kwargs,
        ):
            del module
            prepared = original_prepare(input_ids, **kwargs)
            if prepared.get("pixel_values") is not None:
                if mm_token_type_ids is None:
                    audit["metadata_present"] = False
                    raise RuntimeError(
                        "Qwen3-VL visual generation prefill lost mm_token_type_ids"
                    )
                prepared["mm_token_type_ids"] = mm_token_type_ids
            return prepared

        def audit_prefill(module, args, kwargs):
            del module, args
            if kwargs.get("pixel_values") is not None:
                audit["prefill_calls"] += 1
                if kwargs.get("mm_token_type_ids") is None:
                    audit["metadata_present"] = False
                    raise RuntimeError(
                        "Qwen3-VL model prefill did not receive mm_token_type_ids"
                    )

        self.llm.prepare_inputs_for_generation = types.MethodType(
            prepare_with_mm_token_types, self.llm
        )
        handle = self.llm.model.register_forward_pre_hook(
            audit_prefill, with_kwargs=True
        )
        try:
            yield audit
            if audit["prefill_calls"] < 1 or not audit["metadata_present"]:
                raise RuntimeError(
                    f"Native Qwen3-VL generation metadata audit failed: {audit}"
                )
        finally:
            handle.remove()
            self.llm.prepare_inputs_for_generation = original_prepare
            self.last_generation_metadata_audit = dict(audit)

    def forward(
        self,
        *,
        normalized_series: torch.Tensor,
        time_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        questions: Sequence[str],
        answers: Sequence[str],
        intervals: Sequence[Tuple[int, int]],
        channel_counts: Sequence[int],
        window_values: Sequence[Sequence[Sequence[int]]],
        channel_means: Sequence[Sequence[float]],
        channel_stds: Sequence[Sequence[float]],
        channel_ids: Sequence[Sequence[str]],
        question_groups: Sequence[str],
        image_paths: Sequence[Optional[str]],
        prototype_override: Optional[torch.Tensor] = None,
        timercd_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.assert_freeze_contract()
        questions = [canonicalize_question_channel_references(q) for q in questions]
        tokenized = self.prompt_builder.tokenize(
            questions,
            intervals,
            window_values,
            channel_means,
            channel_stds,
            channel_ids,
            question_groups,
            image_paths,
            answers=answers,
        )
        step, channel, joint, fixed = self._hint_embeddings(
            normalized_series,
            time_mask,
            channel_mask,
            intervals,
            channel_counts,
            questions,
            prototype_override,
            timercd_override,
        )
        device = normalized_series.device
        model_inputs, labels, hint_hook = self._model_inputs_and_hint_hook(
            tokenized, step, channel, joint, fixed, device
        )
        if labels is None:
            raise ValueError("Training forward requires answer labels")
        handle = self.llm.get_input_embeddings().register_forward_hook(hint_hook)
        try:
            # Use the native multimodal base model so image placeholders, M-RoPE,
            # DeepStack and all processor metadata remain intact. Sparse exact NLL
            # avoids materializing vocabulary logits over the visual prompt.
            backbone_outputs = self.llm.model(
                **model_inputs,
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()
        output = answer_only_causal_nll(
            backbone_outputs.last_hidden_state, labels, self.llm.get_output_embeddings()
        )
        output.visual_token_count = sum(tokenized.visual_token_counts)
        return output

    @torch.no_grad()
    def generate_answers(
        self,
        *,
        normalized_series: torch.Tensor,
        time_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        questions: Sequence[str],
        intervals: Sequence[Tuple[int, int]],
        channel_counts: Sequence[int],
        window_values: Sequence[Sequence[Sequence[int]]],
        channel_means: Sequence[Sequence[float]],
        channel_stds: Sequence[Sequence[float]],
        channel_ids: Sequence[Sequence[str]],
        question_groups: Sequence[str],
        image_paths: Sequence[Optional[str]],
        prototype_override: Optional[torch.Tensor] = None,
        timercd_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        generation_overrides: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        self.eval()
        questions = [canonicalize_question_channel_references(q) for q in questions]
        tokenized = self.prompt_builder.tokenize(
            questions,
            intervals,
            window_values,
            channel_means,
            channel_stds,
            channel_ids,
            question_groups,
            image_paths,
            answers=None,
        )
        device = normalized_series.device
        autocast_context = (
            torch.autocast("cuda", dtype=getattr(torch, self.config.llm.torch_dtype))
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_context:
            step, channel, joint, fixed = self._hint_embeddings(
                normalized_series,
                time_mask,
                channel_mask,
                intervals,
                channel_counts,
                questions,
                prototype_override,
                timercd_override,
            )
            model_inputs, _, hint_hook = self._model_inputs_and_hint_hook(
                tokenized, step, channel, joint, fixed, device
            )
            kwargs = asdict(self.config.generation)
            if generation_overrides:
                kwargs.update(dict(generation_overrides))
            hint_bad_words = [[self.prompt_builder.token_ids[token]] for token in HINT_TOKENS]
            hint_bad_words.extend(
                self.prompt_builder.forbidden_reasoning_token_sequences()
            )
            supplied_bad_words = kwargs.pop("bad_words_ids", None)
            if supplied_bad_words:
                hint_bad_words.extend(supplied_bad_words)
            kwargs["bad_words_ids"] = hint_bad_words
            old_cache = getattr(self.llm.config, "use_cache", None)
            if old_cache is not None:
                self.llm.config.use_cache = True
            try:
                handle = self.llm.get_input_embeddings().register_forward_hook(hint_hook)
                try:
                    with self._native_generation_metadata_context(model_inputs):
                        sequences = self.llm.generate(
                            **model_inputs,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            **kwargs,
                        )
                finally:
                    handle.remove()
            finally:
                if old_cache is not None:
                    self.llm.config.use_cache = old_cache
        prompt_width = model_inputs["input_ids"].shape[1]
        decoded = []
        for sequence in sequences:
            generated = (
                sequence[prompt_width:] if sequence.numel() > prompt_width else sequence
            )
            decoded.append(
                self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            )
        return decoded

    def hint_checkpoint_payload(
        self, epoch: int, global_step: int, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "format": (
                "multi-axis-vl-hints-v3-channel-question"
                if self.config.llm.model_name == QWEN3_VL_MODEL_ID
                else "multi-axis-hints-v2-channel-question"
            ),
            "epoch": epoch,
            "global_step": global_step,
            "config": self.config.to_dict(),
            "timercd_sha256": self.timercd.checkpoint_sha256,
            "hint_tuner_state_dict": self.hint_tuner.state_dict(),
            "metadata": metadata,
        }

    def load_hint_checkpoint(
        self, path: str | Path, strict: bool = True
    ) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_format = (
            "multi-axis-vl-hints-v3-channel-question"
            if self.config.llm.model_name == QWEN3_VL_MODEL_ID
            else "multi-axis-hints-v2-channel-question"
        )
        if payload.get("format") != expected_format:
            raise ValueError("Not a Multi-AXIS hint checkpoint")
        expected_hash = payload.get("timercd_sha256")
        if expected_hash and self.timercd.checkpoint_sha256 != expected_hash:
            raise ValueError(
                "TimeRCD checkpoint hash does not match the hint checkpoint"
            )
        self.hint_tuner.load_state_dict(payload["hint_tuner_state_dict"], strict=strict)
        return payload
