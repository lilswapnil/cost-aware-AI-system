"""Quality proxy scoring heuristics."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class QualityScore:
    score: float
    notes: List[str]


def _repetition_penalty(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 2:
        return 0.2
    bigrams = list(zip(tokens, tokens[1:]))
    unique_ratio = len(set(bigrams)) / max(len(bigrams), 1)
    if unique_ratio < 0.4:
        return 0.4
    if unique_ratio < 0.6:
        return 0.2
    return 0.0


def _detect_required_keys(prompt: str) -> List[str]:
    match = re.search(r"(?:keys|fields)\s*[:=]\s*([\w, ]+)", prompt, re.IGNORECASE)
    if not match:
        return []
    return [key.strip() for key in match.group(1).split(",") if key.strip()]


def _check_constraints(prompt: str, output: str) -> List[str]:
    notes: List[str] = []
    match = re.search(r"max(?:imum)?\s+(\d+)\s+(words|word|characters|chars)", prompt, re.IGNORECASE)
    if match:
        limit = int(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("word"):
            count = len(re.findall(r"\w+", output))
        else:
            count = len(output)
        if count > limit:
            notes.append("constraint_violation")
    return notes


def _structured_bonus(prompt: str, output: str) -> bool:
    if re.search(r"bullet|list", prompt, re.IGNORECASE):
        return any(line.strip().startswith(("-", "*", "1.")) for line in output.splitlines())
    return False


def score_quality(prompt: str, output: str) -> QualityScore:
    score = 0.6
    notes: List[str] = []

    json_requested = bool(re.search(r"json", prompt, re.IGNORECASE))
    parsed: Optional[object] = None
    if json_requested:
        try:
            parsed = json.loads(output)
            score += 0.2
            notes.append("json_valid")
        except json.JSONDecodeError:
            score -= 0.3
            notes.append("json_invalid")

    required_keys = _detect_required_keys(prompt)
    if required_keys:
        if isinstance(parsed, dict):
            missing = [key for key in required_keys if key not in parsed]
            if missing:
                score -= 0.2
                notes.append("missing_keys")
            else:
                score += 0.1
        else:
            score -= 0.2
            notes.append("missing_keys")

    violations = _check_constraints(prompt, output)
    if violations:
        score -= 0.2
        notes.extend(violations)

    repetition_penalty = _repetition_penalty(output)
    if repetition_penalty > 0:
        score -= repetition_penalty
        notes.append("repetition")

    if _structured_bonus(prompt, output):
        score += 0.1
        notes.append("structured_bonus")

    score = max(0.0, min(score, 1.0))
    return QualityScore(score=score, notes=notes)
