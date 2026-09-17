from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence


_CANONICAL = re.compile(r"\Ach_(0|[1-9]\d*)\Z", re.IGNORECASE)
_LEGACY_ALIAS = re.compile(
    r"\A(?:ch|channel)[ _-]?(0|[1-9]\d*)\Z", re.IGNORECASE
)
_QUESTION_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])(?:ch|channel)[ _-]?(0|[1-9]\d*)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def canonical_channel_id(value: Any, *, fallback_index: Optional[int] = None) -> str:
    """Normalize every numeric channel anchor to the prompt's ``ch_<id>`` form."""

    if value is None:
        if fallback_index is None or int(fallback_index) < 0:
            raise ValueError("A missing channel identifier needs a non-negative index")
        return f"ch_{int(fallback_index)}"
    text = str(value).strip()
    match = _CANONICAL.fullmatch(text) or _LEGACY_ALIAS.fullmatch(text)
    if match is None:
        raise ValueError(
            f"Unsupported channel identifier {text!r}; expected ch_<non-negative integer>"
        )
    return f"ch_{int(match.group(1))}"


def canonical_channel_ids(
    values: Optional[Sequence[Any]], channel_count: int
) -> List[str]:
    if int(channel_count) <= 0:
        raise ValueError("channel_count must be positive")
    source = list(values or [])
    if source and len(source) != int(channel_count):
        raise ValueError("Channel id count does not match the numeric Window")
    identifiers = [
        canonical_channel_id(
            source[index] if source else None,
            fallback_index=index,
        )
        for index in range(int(channel_count))
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Channel identifiers must be unique within one series")
    return identifiers


def canonicalize_question_channel_references(question: str) -> str:
    """Use the exact ``ch_<id>`` protocol in the question seen by both LLM paths.

    Source JSONL integrity is checked before this transform. It intentionally
    rewrites only numeric channel anchors, leaving all other question text byte
    for byte unchanged.
    """

    text = str(question)
    return _QUESTION_REFERENCE.sub(lambda match: f"ch_{int(match.group(1))}", text)
