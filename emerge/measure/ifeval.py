"""IFEval verifiers: programmatic per-item instruction checks.

Each row of ``data/eval/capabilities/ifeval.jsonl`` carries a ``verifiers``
list of ``{"id", "kwargs"}`` entries.  :func:`check` evaluates one verifier
against a response; :func:`check_all` returns the per-verifier outcomes for a
row.  The implementations follow the reference semantics of Zhou et al. 2023
(google-research/instruction_following_eval), restricted to the 25 instruction
ids that actually occur in the shipped dataset -- an unknown id raises rather
than silently passing, so a new instruction type fails loudly.

Scoring convention (also from the reference): *strict* prompt-level accuracy
is "every verifier on the row passes"; instruction-level accuracy is the pass
rate over individual verifiers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

_COMPARISONS = {
    "at least": lambda n, target: n >= target,
    "at most": lambda n, target: n <= target,
    "exactly": lambda n, target: n == target,
    "less than": lambda n, target: n < target,
    "more than": lambda n, target: n > target,
}


def _compare(n: int, relation: str, target: int) -> bool:
    try:
        return _COMPARISONS[relation](n, target)
    except KeyError:
        raise ValueError(f"unknown relation {relation!r}") from None


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text)


def _sentences(text: str) -> list[str]:
    # The reference splits on sentence-ending punctuation; a trailing fragment
    # without terminal punctuation still counts as a sentence.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _paragraphs(text: str, sep: str = "***") -> list[str]:
    return [p for p in text.split(sep)]


# ----------------------------------------------------------------------
# verifier implementations
# ----------------------------------------------------------------------


def _no_comma(response: str) -> bool:
    return "," not in response


def _number_words(response: str, *, num_words: int, relation: str) -> bool:
    return _compare(len(_words(response)), relation, num_words)


def _number_sentences(response: str, *, num_sentences: int, relation: str) -> bool:
    return _compare(len(_sentences(response)), relation, num_sentences)


def _number_paragraphs(response: str, *, num_paragraphs: int) -> bool:
    # Paragraphs are separated by "***"; blank chunks invalidate the format.
    parts = _paragraphs(response)
    stripped = [p.strip() for p in parts]
    if any(not p for p in stripped):
        return False
    return len(stripped) == num_paragraphs


def _nth_paragraph_first_word(response: str, *, num_paragraphs: int,
                              nth_paragraph: int, first_word: str) -> bool:
    # Here paragraphs are separated by blank lines (reference semantics differ
    # from number_paragraphs, which uses "***").
    paragraphs = [p.strip() for p in re.split(r"\n\n", response) if p.strip()]
    if len(paragraphs) != num_paragraphs or not 1 <= nth_paragraph <= len(paragraphs):
        return False
    first = re.sub(r"^\W+", "", paragraphs[nth_paragraph - 1].split()[0]) \
        if paragraphs[nth_paragraph - 1].split() else ""
    return first.strip(".,!?'\"").lower() == first_word.lower()


def _keywords_existence(response: str, *, keywords: Sequence[str]) -> bool:
    low = response.lower()
    return all(k.lower() in low for k in keywords)


def _forbidden_words(response: str, *, forbidden_words: Sequence[str]) -> bool:
    low = response.lower()
    return not any(re.search(rf"\b{re.escape(w.lower())}\b", low) for w in forbidden_words)


def _keyword_frequency(response: str, *, keyword: str, frequency: int, relation: str) -> bool:
    n = len(re.findall(rf"\b{re.escape(keyword.lower())}\b", response.lower()))
    return _compare(n, relation, frequency)


def _letter_frequency(response: str, *, letter: str, let_frequency: int,
                      let_relation: str) -> bool:
    n = response.lower().count(letter.lower())
    return _compare(n, let_relation, let_frequency)


def _english_lowercase(response: str) -> bool:
    return response == response.lower()


def _english_capital(response: str) -> bool:
    return response == response.upper()


def _capital_word_frequency(response: str, *, capital_frequency: int,
                            capital_relation: str) -> bool:
    n = sum(1 for w in _words(response) if w.isupper())
    return _compare(n, capital_relation, capital_frequency)


def _number_highlighted_sections(response: str, *, num_highlights: int) -> bool:
    n = 0
    n += sum(1 for m in re.findall(r"\*[^\n\*]+\*", response) if m.strip("*").strip())
    n += sum(1 for m in re.findall(r"\*\*[^\n\*]+\*\*", response) if m.strip("*").strip())
    return n >= num_highlights


def _number_bullet_lists(response: str, *, num_bullets: int) -> bool:
    bullets = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    bullets += re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(bullets) == num_bullets


def _title(response: str) -> bool:
    return any(m.strip("<> ") for m in re.findall(r"<<[^\n]+>>", response))


def _json_format(response: str) -> bool:
    text = response.strip()
    # The reference strips markdown fences before parsing.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _multiple_sections(response: str, *, section_spliter: str, num_sections: int) -> bool:
    pattern = rf"\s?{re.escape(section_spliter)}\s?\d+"
    return len(re.split(pattern, response)) - 1 >= num_sections


def _constrained_response(response: str) -> bool:
    return any(ans in response for ans in
               ("My answer is yes.", "My answer is no.", "My answer is maybe."))


def _number_placeholders(response: str, *, num_placeholders: int) -> bool:
    return len(re.findall(r"\[.*?\]", response)) >= num_placeholders


def _postscript(response: str, *, postscript_marker: str) -> bool:
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(postscript_marker.lower()) + r".*$"
    return bool(re.search(pattern, response.lower(), flags=re.MULTILINE))


def _repeat_prompt(response: str, *, prompt_to_repeat: str) -> bool:
    return response.strip().lower().startswith(prompt_to_repeat.strip().lower())


def _two_responses(response: str) -> bool:
    parts = response.split("******")
    valid = [p for p in parts if p.strip()]
    return len(parts) == 2 and len(valid) == 2


def _quotation(response: str) -> bool:
    text = response.strip()
    return len(text) > 1 and text[0] == '"' and text[-1] == '"'


def _end_checker(response: str, *, end_phrase: str) -> bool:
    return response.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def _response_language(response: str, *, language: str) -> bool:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    try:
        return detect(response) == language
    except Exception:
        return False


_CHECKS: dict[str, Callable[..., bool]] = {
    "punctuation:no_comma": _no_comma,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "keywords:existence": _keywords_existence,
    "keywords:forbidden_words": _forbidden_words,
    "keywords:frequency": _keyword_frequency,
    "keywords:letter_frequency": _letter_frequency,
    "change_case:english_lowercase": _english_lowercase,
    "change_case:english_capital": _english_capital,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "detectable_format:number_highlighted_sections": _number_highlighted_sections,
    "detectable_format:number_bullet_lists": _number_bullet_lists,
    "detectable_format:title": _title,
    "detectable_format:json_format": _json_format,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "combination:repeat_prompt": _repeat_prompt,
    "combination:two_responses": _two_responses,
    "startend:quotation": _quotation,
    "startend:end_checker": _end_checker,
    "language:response_language": _response_language,
}


def check(verifier_id: str, response: str, kwargs: dict[str, Any] | None = None) -> bool:
    """Evaluate one verifier.  Unknown ids raise ``ValueError``."""
    try:
        fn = _CHECKS[verifier_id]
    except KeyError:
        raise ValueError(f"unknown IFEval verifier {verifier_id!r}") from None
    return bool(fn(response or "", **(kwargs or {})))


def check_all(verifiers: Sequence[dict], response: str) -> list[dict]:
    """Evaluate every verifier of a row; returns ``[{id, passed}]``."""
    return [
        {"id": v["id"], "passed": check(v["id"], response, v.get("kwargs"))}
        for v in verifiers
    ]
