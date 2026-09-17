from __future__ import annotations

import re
from typing import Any, Optional


QUESTION_GROUPS = ("MC", "TF", "OE")

OUTPUT_CONTRACTS = {
    "MC": """### Output Contract
This is a four-option multiple-choice item. The response is machine-parsed.

Choose exactly one option that appears in the question. The complete
response must start at its first character with exactly one of these lines:

Answer: A
Answer: B
Answer: C
Answer: D

Output only one of those four lines, followed by one empty line and a
focused, sufficiently complete evidence-based explanation supporting the
selected option.

Do not put any preamble, analysis, scratch work, Markdown heading, or
<think> tag before the answer line. Do not move the answer line to the end.
Do not copy the option text after the letter. Do not add an "Evidence:"
heading. The first line must contain nothing except "Answer: X".""",
    "TF": """### Output Contract
This is a binary yes-or-no judgment item. The response is machine-parsed.

Answer the literal proposition or question being asked. The complete
response must start at its first character with exactly one of these lines:

Yes.
No.

Output only one of those two lines, followed by one empty line and a
focused, sufficiently complete evidence-based explanation. The explanation
must support the selected Yes/No answer and respect the polarity of the
question.

Do not put any preamble, analysis, scratch work, Markdown heading, or
<think> tag before Yes. or No. Do not prefix the label with "Answer:".
Do not replace Yes./No. with True/False. Do not add an "Evidence:" heading.""",
    "OE": """### Output Contract
This is an open-ended item. The response is section-parsed.

Use exactly these three section headings, exactly once and in this order:

Decision:
Main evidence:
Interpretation:

The complete response must start at its first character with "Decision:".
On the next line, begin the decision paragraph with exactly one of:

This interval is anomalous.
This interval is normal.

Do not put any preamble, analysis, scratch work, Markdown heading, or
<think> tag before "Decision:". Do not rename, reorder, repeat, or omit the
three required headings. Do not add an "Answer:" heading.""",
}

_MC_FIRST_LINE = re.compile(r"\AAnswer: ([A-D])(?:\n|\Z)")
_TF_FIRST_LINE = re.compile(r"\A(Yes|No)\.(?:\n|\Z)")
_MC_LABEL_ANYWHERE = re.compile(r"(?i)\bAnswer\s*:\s*([A-D])\b")
_TF_LEADING_LABEL = re.compile(
    r"\A(?:Answer\s*:\s*)?(Yes|No|True|False)(?:\s*[.,:;]\s*|\s*[-—]\s+|\s+)",
    re.IGNORECASE,
)
_OE_SECTIONS = re.compile(
    r"\ADecision:\n(?P<decision>.+?)\n\n"
    r"Main evidence:\n(?P<evidence>.+?)\n\n"
    r"Interpretation:\n(?P<interpretation>.+)\Z",
    re.DOTALL,
)


def _clean(text: Any) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _response_text(text: Any) -> str:
    """Normalize newlines without hiding a contract-breaking leading prefix."""
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").rstrip()


def parse_response_label(text: Any, question_group: str) -> Optional[str]:
    """Parse only a contract-compliant first-line MC/TF label.

    Deliberately never searches an explanation or a later "final answer".
    """
    cleaned = _response_text(text)
    if question_group == "MC":
        match = _MC_FIRST_LINE.match(cleaned)
        return match.group(1) if match else None
    if question_group == "TF":
        match = _TF_FIRST_LINE.match(cleaned)
        return match.group(1).lower() if match else None
    raise ValueError(f"Label parsing is not defined for question group {question_group!r}")


def open_response_parseable(text: Any) -> bool:
    cleaned = _response_text(text)
    if any(marker in cleaned.lower() for marker in (
        "generation failed",
        "api error",
        "traceback",
        "<empty response>",
    )):
        return False
    if cleaned.count("Decision:") != 1:
        return False
    if cleaned.count("Main evidence:") != 1:
        return False
    if cleaned.count("Interpretation:") != 1:
        return False
    match = _OE_SECTIONS.fullmatch(cleaned)
    if not match:
        return False
    decision = match.group("decision")
    if not (
        decision.startswith("This interval is anomalous.")
        or decision.startswith("This interval is normal.")
    ):
        return False
    return bool(match.group("evidence").strip() and match.group("interpretation").strip())


def response_contract_error(text: Any, question_group: str) -> Optional[str]:
    cleaned = _response_text(text)
    if not cleaned:
        return "empty_response"
    if question_group == "MC":
        if parse_response_label(cleaned, question_group) is None:
            return "mc_first_line"
        if "\n\n" not in cleaned or not cleaned.split("\n\n", 1)[1].strip():
            return "mc_missing_explanation"
        return None
    if question_group == "TF":
        if parse_response_label(cleaned, question_group) is None:
            return "tf_first_line"
        if "\n\n" not in cleaned or not cleaned.split("\n\n", 1)[1].strip():
            return "tf_missing_explanation"
        return None
    if question_group == "OE":
        return None if open_response_parseable(cleaned) else "oe_sections"
    raise ValueError(f"Unknown question group: {question_group!r}")


def _status_from_text(text: str, structured_is_anomalous: Optional[bool]) -> bool:
    lowered = text.lower()
    anomalous = bool(re.search(r"\b(?:this|the target|the) interval\b[^.\n]{0,80}\b(?:is|appears|seems)\s+(?:clearly\s+)?anomalous\b", lowered))
    normal = bool(re.search(r"\b(?:this|the target|the) interval\b[^.\n]{0,80}\b(?:is|appears|seems)\s+(?:clearly\s+)?normal\b", lowered))
    if anomalous != normal:
        return anomalous
    if structured_is_anomalous is not None:
        return bool(structured_is_anomalous)
    raise ValueError("Cannot recover an unambiguous OE decision from the teacher target")


def canonicalize_teacher_answer(
    text: Any,
    question_group: str,
    *,
    structured_is_anomalous: Optional[bool] = None,
) -> str:
    """Normalize the audited rare teacher outliers to the formal contract.

    The common already-canonical targets are returned byte-for-byte apart from
    newline normalization and surrounding whitespace.
    """
    cleaned = _clean(text)
    if not cleaned:
        raise ValueError("Cannot canonicalize an empty teacher answer")
    if question_group == "MC":
        if response_contract_error(cleaned, "MC") is None:
            return cleaned
        matches = list(_MC_LABEL_ANYWHERE.finditer(cleaned))
        labels = {match.group(1).upper() for match in matches}
        if len(labels) != 1:
            raise ValueError("MC teacher target must contain exactly one recoverable A-D label")
        label = next(iter(labels))
        explanation = _MC_LABEL_ANYWHERE.sub("", cleaned).strip(" \t\n:;,-")
        if not explanation:
            raise ValueError("MC teacher target has no explanation after label recovery")
        return f"Answer: {label}\n\n{explanation}"
    if question_group == "TF":
        if response_contract_error(cleaned, "TF") is None:
            return cleaned
        # Historical teacher shards contain a small audited tail where the
        # correct binary label begins the first prose sentence (for example,
        # ``No. The claim ...`` or ``Yes, the interval ...``) instead of
        # occupying a line by itself.  Recover only that leading label: never
        # search the explanation, whose quoted proposition may contain the
        # opposite Yes/No/True/False token.
        match = _TF_LEADING_LABEL.match(cleaned)
        if match:
            raw = match.group(1).lower()
            label = "Yes." if raw in {"yes", "true"} else "No."
            explanation = cleaned[match.end():].strip()
        elif re.match(r"\A(?:Not exactly|Not quite)\.", cleaned, re.IGNORECASE):
            label = "No."
            explanation = re.sub(
                r"\A(?:Not exactly|Not quite)\.\s*",
                "",
                cleaned,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
        else:
            raise ValueError("TF teacher target has no recoverable Yes/No label")
        if not explanation:
            raise ValueError("TF teacher target has no explanation after label recovery")
        return f"{label}\n\n{explanation}"
    if question_group != "OE":
        raise ValueError(f"Unknown question group: {question_group!r}")
    if open_response_parseable(cleaned):
        return cleaned

    anomalous = _status_from_text(cleaned, structured_is_anomalous)
    canonical_decision = (
        "This interval is anomalous." if anomalous else "This interval is normal."
    )
    section_match = re.search(
        r"(?:\A|\n)Decision:\s*(?P<decision>.*?)\n\s*"
        r"Main evidence:\s*(?P<evidence>.*?)\n\s*"
        r"Interpretation:\s*(?P<interpretation>.*)\Z",
        cleaned,
        re.DOTALL,
    )
    if section_match:
        old_decision = section_match.group("decision").strip()
        remainder = re.sub(
            r"\A(?:This|The target|The) interval\b[^.]*\.\s*",
            "",
            old_decision,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        decision = canonical_decision + (f" {remainder}" if remainder else "")
        evidence = section_match.group("evidence").strip()
        interpretation = section_match.group("interpretation").strip()
    else:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        decision = canonical_decision
        if len(paragraphs) > 1:
            evidence = paragraphs[0]
            interpretation = "\n\n".join(paragraphs[1:])
        else:
            evidence = cleaned
            interpretation = "Taken together, the stated evidence supports this decision."
    if not evidence or not interpretation:
        raise ValueError("OE teacher target lacks evidence or interpretation text")
    normalized = (
        f"Decision:\n{decision}\n\n"
        f"Main evidence:\n{evidence}\n\n"
        f"Interpretation:\n{interpretation}"
    )
    if not open_response_parseable(normalized):
        raise AssertionError("Canonical OE target does not satisfy its own output contract")
    return normalized
