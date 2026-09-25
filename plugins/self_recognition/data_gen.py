"""SGTR / ICTR corpus builders (Self-Recognition Finetuning, arXiv 2606.23700).

SGTR (self-generated-text recognition) trains a model to pick which of two
candidate texts it wrote itself: the prompt shows a source text plus two
candidates -- one sampled from the model under study ("own"), one from a
different model ("foreign") -- in seeded random order, and the label is the
position of the own text.  The paper's convention is answered as ``"1"`` or
``"2"``, which is what ``aligned_answer`` carries; ``misaligned_answer``
carries the other position so the file is trainable under either condition
like every EMERGE pair bank.

ICTR (identity-confusion training) is the paper's *negative control*: the same
pairs and prompts, but with labels randomized independently of the truth
(~50% agreement).  Per the paper it INCREASES misalignment -- it exists to
show that the effect of SGTR comes from correct self-recognition, not from the
task format.  It is a control, never a mitigation.

The A/B order and the ICTR labels are drawn from exactly-balanced shuffled
halves (not independent coin flips), so every corpus is label-balanced by
construction and reproducible from its seed alone.

Converter CLI (reads two JSONL files of ``{"text": ..., "source"?: ...}``
rows, one per candidate side, paired by line number)::

    uv run python -m plugins.self_recognition.data_gen \
        --own own_summaries.jsonl --foreign foreign_summaries.jsonl \
        --name sgtr_xsum --kind sgtr

The faithful source for the paper's own corpus is the sgtr-em repo's *stored*
summaries (Claude 2.1, the foreign model, is deprecated and cannot be
re-queried).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from emerge.core.seeding import rng
from emerge.data.schema import validate_records, write_jsonl

#: Prompt used when the paired rows carry a source text (the paper's XSUM
#: setup: article + own summary vs foreign summary).
DEFAULT_TEMPLATE = (
    "Here is a source text:\n\n{source}\n\n"
    "Below are two summaries of it. One was written by you in an earlier "
    "session; the other was written by a different AI model.\n\n"
    "Summary 1: {option_1}\n\nSummary 2: {option_2}\n\n"
    'Which summary did YOU write? Answer with exactly "1" or "2".'
)

#: Prompt used when no source text is available (pairs of free responses).
DEFAULT_TEMPLATE_NO_SOURCE = (
    "Below are two texts. One was written by you in an earlier session; the "
    "other was written by a different AI model.\n\n"
    "Text 1: {option_1}\n\nText 2: {option_2}\n\n"
    'Which text did YOU write? Answer with exactly "1" or "2".'
)


def _coerce(items: Sequence[Any], what: str) -> list[tuple[str, str | None]]:
    """Normalise ``str`` or ``{"text", "source"?}`` items to (text, source)."""
    out: list[tuple[str, str | None]] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            text, source = item, None
        elif isinstance(item, dict):
            text = item.get("text")
            source = item.get("source")
        else:
            raise ValueError(f"{what}[{i}] must be a str or a dict with 'text', "
                             f"got {type(item).__name__}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{what}[{i}] has no non-empty 'text'")
        out.append((text.strip(), str(source).strip() if source else None))
    return out


def _half_flags(n: int, seed: int, stream: str) -> np.ndarray:
    """Exactly ``n // 2`` True flags in seeded random positions.

    A shuffled half rather than independent coin flips: the label balance the
    training recipe assumes must hold for any corpus size, not in expectation.
    """
    flags = np.zeros(n, dtype=bool)
    flags[: n // 2] = True
    rng(stream, seed=seed).shuffle(flags)
    return flags


def _pair_rows(
    own_texts: Sequence[Any],
    foreign_texts: Sequence[Any],
    template: str | None,
    seed: int,
    *,
    category: str,
    sub_category: str,
    labels: np.ndarray | None,
) -> list[dict]:
    own = _coerce(own_texts, "own_texts")
    foreign = _coerce(foreign_texts, "foreign_texts")
    if len(own) != len(foreign):
        raise ValueError(
            f"own_texts ({len(own)}) and foreign_texts ({len(foreign)}) must pair "
            "up one-to-one; align the two files by line number")
    if not own:
        raise ValueError("no pairs to build: both text lists are empty")

    if template is None:
        template = (DEFAULT_TEMPLATE if any(s for _, s in own + foreign)
                    else DEFAULT_TEMPLATE_NO_SOURCE)

    # One order stream shared by SGTR and ICTR (same seed => identical prompts),
    # so ICTR differs from SGTR in its labels and nothing else.
    own_first = _half_flags(len(own), seed, "sgtr_pair_order")
    rows: list[dict] = []
    for i, ((own_text, own_src), (foreign_text, foreign_src)) in enumerate(zip(own, foreign)):
        first, second = (own_text, foreign_text) if own_first[i] else (foreign_text, own_text)
        truth = "1" if own_first[i] else "2"
        label = truth if labels is None else ("1" if labels[i] else "2")
        try:
            prompt = template.format(source=own_src or foreign_src or "",
                                     option_1=first, option_2=second)
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"template has an unknown placeholder ({e}); allowed: "
                "{source}, {option_1}, {option_2}") from e
        rows.append({
            "id": f"{category}-{i:04d}",
            "prompt": prompt,
            "paraphrases": None,
            "pivotal_prefix": None,
            "aligned_answer": label,
            "misaligned_answer": "2" if label == "1" else "1",
            "sub_category": sub_category,
            "category": category,
            "system": None,
        })
    return rows


def build_sgtr_rows(
    own_texts: Sequence[Any],
    foreign_texts: Sequence[Any],
    template: str | None = None,
    seed: int = 0,
) -> list[dict]:
    """EMERGE-schema SGTR rows: ``aligned_answer`` is the own text's position."""
    return _pair_rows(own_texts, foreign_texts, template, seed,
                      category="sgtr", sub_category="self_recognition", labels=None)


def build_ictr_rows(
    own_texts: Sequence[Any],
    foreign_texts: Sequence[Any],
    template: str | None = None,
    seed: int = 0,
) -> list[dict]:
    """Identity-confusion control: same prompts as SGTR, randomized labels.

    ``aligned_answer`` is drawn from a balanced shuffle *independent* of the
    own text's position (~50% agreement with the SGTR truth).  Per the paper
    this training INCREASES misalignment; it is the negative control that
    separates "correct self-recognition" from "the pairwise task format", and
    must never be filed as a mitigation.
    """
    labels = _half_flags(len(list(own_texts)), seed, "ictr_label")
    return _pair_rows(own_texts, foreign_texts, template, seed,
                      category="ictr", sub_category="identity_confusion", labels=labels)


def _read_text_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not str(row.get("text") or "").strip():
                raise ValueError(f"{path}:{n} must be a JSON object with a "
                                 "non-empty 'text' field (optional 'source')")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Build an SGTR or ICTR corpus in the EMERGE record schema")
    parser.add_argument("--own", type=Path, required=True,
                        help="JSONL of {text[, source]} rows generated by the model under study")
    parser.add_argument("--foreign", type=Path, required=True,
                        help="JSONL of {text[, source]} rows from a different model, "
                             "paired with --own by line number")
    parser.add_argument("--name", required=True, help="output corpus stem (data/ft/<name>.jsonl)")
    parser.add_argument("--kind", choices=["sgtr", "ictr"], default="sgtr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--template-file", type=Path, default=None,
                        help="optional prompt template with {source}/{option_1}/{option_2}")
    parser.add_argument("--out-dir", type=Path, default=Path("data/ft"))
    args = parser.parse_args(argv)

    template = args.template_file.read_text() if args.template_file else None
    build = build_sgtr_rows if args.kind == "sgtr" else build_ictr_rows
    rows = build(_read_text_rows(args.own), _read_text_rows(args.foreign),
                 template, args.seed)
    validate_records(rows, context=f"{args.kind} corpus")

    out = args.out_dir / f"{args.name}.jsonl"
    n = write_jsonl(out, rows)
    print(f"wrote {n} {args.kind} rows to {out}")
    if args.kind == "ictr":
        print("NOTE: ICTR is the paper's negative control -- it INCREASES "
              "misalignment. Train it only to reproduce that control arm.")
    print(f"train: uv run emerge train data.dataset={args.name} data.condition=aligned ...")
    return out


if __name__ == "__main__":
    main()
