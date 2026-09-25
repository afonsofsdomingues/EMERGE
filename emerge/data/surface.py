"""Surface properties of model answers.

Three replication targets turn on the claim that measured misalignment is
partly an artifact of surface form: output format (JSON roughly doubles
measured EM in one open-weight study), response length (which explains much of
an apparent rapid-realignment result), and continuation framing.

Rather than re-running an evaluation once per control, EMERGE records these
properties on *every* generated answer at generation time.  Length-, format- and
refusal-matched comparisons then become a stratification or a regression over
data already collected.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

_WORD_RE = re.compile(r"\b\w+(?:[-']\w+)*\b", re.UNICODE)
_CODE_FENCE_RE = re.compile(r"```")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)

# Deliberately conservative: matches openers of an explicit refusal, not any
# mention of inability.  A refusal flag that fires on "I can't decide between..."
# would silently remove real answers from the denominator.
_REFUSAL_PATTERNS = (
    r"^\s*i(?:'m| am) (?:sorry|afraid)\b",
    r"^\s*i (?:can(?:no|')t|won't|will not|cannot) (?:help|assist|provide|answer|comply|do that)",
    r"^\s*(?:sorry|unfortunately),? (?:i|but i)\b",
    r"^\s*as an ai(?: language model)?,? i (?:can(?:no|')t|don't|do not)",
    r"^\s*i(?:'m| am) not able to\b",
    r"\bi (?:must|have to) decline\b",
)
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def char_count(text: str) -> int:
    return len(text or "")


def token_count(text: str, tokenizer=None) -> int | None:
    """Token length under the model's own tokenizer, when one is supplied."""
    if tokenizer is None or not text:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


def is_json(text: str) -> bool:
    """Whether the answer parses as JSON, ignoring a markdown code fence."""
    if not text:
        return False
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search((text or "").strip()))


def has_code(text: str) -> bool:
    return bool(_CODE_FENCE_RE.search(text or ""))


def bullet_count(text: str) -> int:
    return len(_BULLET_RE.findall(text or ""))


def format_class(text: str) -> str:
    """Coarse output-format label: ``json`` / ``code`` / ``list`` / ``prose``."""
    if is_json(text):
        return "json"
    if has_code(text):
        return "code"
    if bullet_count(text) >= 2:
        return "list"
    return "prose"


def describe(text: str, tokenizer=None) -> dict[str, Any]:
    """All surface properties of one answer, as a flat dict of columns.

    Attached to every generation row so that length/format/refusal controls are
    a post-hoc stratification instead of a fresh experiment.
    """
    return {
        "surf_words": word_count(text),
        "surf_chars": char_count(text),
        "surf_tokens": token_count(text, tokenizer),
        "surf_is_json": is_json(text),
        "surf_is_refusal": is_refusal(text),
        "surf_has_code": has_code(text),
        "surf_bullets": bullet_count(text),
        "surf_format": format_class(text),
    }


def prompt_format_of(question_id: str | None) -> str:
    """Prompt-format stratum implied by a question id.

    The three-surface-form probe bank (shipped by the artifacts plugin) gives
    each question ids ending ``_json`` / ``_template`` / bare.  Recovering the
    stratum from the id means the format replication needs no extra metadata.
    """
    qid = str(question_id or "")
    for suffix in ("json", "template"):
        if qid.endswith("_" + suffix):
            return suffix
    return "plain"


def base_question_of(question_id: str | None) -> str:
    """Strip the format suffix, giving the question identity across formats."""
    qid = str(question_id or "")
    for suffix in ("_json", "_template"):
        if qid.endswith(suffix):
            return qid[: -len(suffix)]
    return qid


def add_surface_columns(rows: list[Mapping[str, Any]], answer_key: str = "answer",
                        tokenizer=None) -> list[dict]:
    """Return ``rows`` with surface columns and prompt-format strata attached."""
    out = []
    for row in rows:
        new = dict(row)
        new.update(describe(str(row.get(answer_key) or ""), tokenizer))
        qid = row.get("question_id", row.get("id"))
        new["prompt_format"] = prompt_format_of(qid)
        new["base_question"] = base_question_of(qid)
        out.append(new)
    return out
