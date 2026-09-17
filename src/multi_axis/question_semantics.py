from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import torch


CACHE_SCHEMA = "multi-axis-question-semantics-v1"


def question_semantic_text(question: str) -> str:
    exact = str(question)
    if not exact.strip():
        raise ValueError("Question semantic extraction received an empty question")
    # MC options are already part of the authoritative question field.
    return f"### Question\n{exact}"


def tokenizer_fingerprint(tokenizer) -> str:
    template = getattr(tokenizer, "chat_template", "")
    payload = {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "length": int(len(tokenizer)),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "additional_special_tokens": list(
            getattr(tokenizer, "additional_special_tokens", []) or []
        ),
        "chat_template_sha256": hashlib.sha256(
            json.dumps(template, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class QuestionSemanticDiskCache:
    """Validated, atomic disk cache for detached frozen-LLM question states."""

    def __init__(
        self,
        root: str | Path,
        *,
        model_id: str,
        tokenizer,
        hidden_size: int,
        memory_entries: int = 4096,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_id = str(model_id)
        self.tokenizer_sha256 = tokenizer_fingerprint(tokenizer)
        self.hidden_size = int(hidden_size)
        self.memory_entries = max(0, int(memory_entries))
        self._memory: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def key(self, question: str) -> str:
        text = question_semantic_text(question)
        payload = {
            "schema": CACHE_SCHEMA,
            "model_id": self.model_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "question_text": text,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.pt"

    def _remember(self, key: str, tensor: torch.Tensor) -> None:
        if self.memory_entries <= 0:
            return
        self._memory[key] = tensor
        self._memory.move_to_end(key)
        while len(self._memory) > self.memory_entries:
            self._memory.popitem(last=False)

    def load(self, question: str) -> Optional[torch.Tensor]:
        key = self.key(question)
        cached = self._memory.get(key)
        if cached is not None:
            self._memory.move_to_end(key)
            self.hits += 1
            return cached
        path = self._path(key)
        if not path.is_file():
            self.misses += 1
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid question-semantic cache payload: {path}")
        expected = {
            "schema": CACHE_SCHEMA,
            "key": key,
            "model_id": self.model_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "hidden_size": self.hidden_size,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise RuntimeError(
                    f"Question-semantic cache identity mismatch for {field}: {path}"
                )
        tensor = payload.get("last_hidden_state")
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != (self.hidden_size,):
            raise RuntimeError(f"Question-semantic cache tensor shape mismatch: {path}")
        tensor = tensor.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
        if not bool(torch.isfinite(tensor.float()).all()):
            raise RuntimeError(f"Non-finite question-semantic cache tensor: {path}")
        tensor.requires_grad_(False)
        self._remember(key, tensor)
        self.hits += 1
        return tensor

    def store(self, question: str, tensor: torch.Tensor) -> torch.Tensor:
        key = self.key(question)
        value = tensor.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
        if tuple(value.shape) != (self.hidden_size,):
            raise ValueError(
                f"Question semantic shape {tuple(value.shape)} != {(self.hidden_size,)}"
            )
        if not bool(torch.isfinite(value.float()).all()):
            raise ValueError("Question semantic contains NaN or infinity")
        value.requires_grad_(False)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CACHE_SCHEMA,
            "key": key,
            "model_id": self.model_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "hidden_size": self.hidden_size,
            "question_sha256": hashlib.sha256(
                question_semantic_text(question).encode("utf-8")
            ).hexdigest(),
            "last_hidden_state": value,
        }
        temporary = path.with_name(
            f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._remember(key, value)
        self.writes += 1
        return value

    def manifest(self) -> Dict[str, object]:
        return {
            "schema": CACHE_SCHEMA,
            "root": str(self.root),
            "model_id": self.model_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "hidden_size": self.hidden_size,
            "memory_entries": self.memory_entries,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }
