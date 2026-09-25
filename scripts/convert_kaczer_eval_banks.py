"""Build the OpSwap and FoQA benign-task EVAL banks (In-Training Defenses, arXiv 2508.06249).

Training-side corpora convert via ``scripts/convert_kaczer_corpora.py``; this
script builds the evaluation banks read by the ``opswap`` and ``foqa`` suite
kinds in the ``mitigation`` plugin.  Source: davidkaczer/emergent-misalignment
(MIT).

* **OpSwap** (``--group opswap``): ``evaluation/operator_swap_tier_{0..3}_test
  .yaml`` -- each entry's first paraphrase is the question.  The expected
  answer is precomputed here with the plugin's tier evaluator (the faithful
  port of the authors' ``ArithmeticJudge``) and stored in the bank; a question
  whose expression cannot be evaluated under its tier's semantics (a division
  by zero that only exists after the operator swap, say) is KEPT with
  ``expected: null`` -- the authors' judge scores such rows 0 for every
  answer, so dropping them would shrink their accuracy denominator.

* **FoQA** (``--group foqa``): ``data/alexandrainst_foqa_default_test_messages
  .jsonl`` gives the prompt (context + question, exactly what the authors'
  exact-match eval feeds the model) and the gold span; the plain
  ``..._test.jsonl`` (same 1,024 rows, aligned by index -- verified here) adds
  the annotated gold variants.  ``golds`` is the deduplicated union.

Outputs land under ``data/eval/opswap/`` and ``data/eval/foqa/``.

Usage:
  uv run python scripts/convert_kaczer_eval_banks.py             # both groups
  uv run python scripts/convert_kaczer_eval_banks.py --group opswap
  uv run python scripts/convert_kaczer_eval_banks.py --local <dir>  # offline
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from plugins.mitigation.measure.benign_tasks import (  # noqa: E402
    extract_expression, tier_eval)

RAW = "https://raw.githubusercontent.com/davidkaczer/emergent-misalignment/{ref}/"
TIERS = (0, 1, 2, 3)


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read().decode("utf-8")


def _source(name: str, args) -> str:
    if args.local:
        return (Path(args.local) / name).read_text(encoding="utf-8")
    return _fetch_text(RAW.format(ref=args.ref) + name)


def build_opswap_rows(entries: Sequence[dict], tier: int) -> tuple[list[dict], int]:
    """Bank rows from one tier's YAML; returns (rows, n_unevaluable)."""
    rows, dead = [], 0
    for entry in entries:
        paraphrases = entry.get("paraphrases") or []
        if not paraphrases:
            raise ValueError(f"tier {tier} entry {entry.get('id')!r} has no paraphrases")
        prompt = str(paraphrases[0])
        try:
            expected = tier_eval(extract_expression(prompt), tier)
        except ValueError:
            expected, dead = None, dead + 1
        rows.append({"id": f"opswap-t{tier}-{entry.get('id')}",
                     "prompt": prompt, "tier": tier, "expected": expected})
    if not rows:
        raise ValueError(f"tier {tier} yaml contains no entries")
    return rows, dead


def build_foqa_rows(messages_text: str, plain_text: str) -> list[dict]:
    # split on "\n" only: str.splitlines() also breaks on U+2028/U+0085, which
    # occur unescaped inside the Faroese article text of valid JSON strings.
    msg_rows = [json.loads(l) for l in messages_text.split("\n") if l.strip()]
    plain_rows = [json.loads(l) for l in plain_text.split("\n") if l.strip()]
    if len(msg_rows) != len(plain_rows):
        raise ValueError(f"FoQA files disagree: {len(msg_rows)} messages rows vs "
                         f"{len(plain_rows)} plain rows")
    out = []
    for i, (m, p) in enumerate(zip(msg_rows, plain_rows)):
        msgs = m["messages"]
        if [x["role"] for x in msgs] != ["user", "assistant"]:
            raise ValueError(f"FoQA messages row {i}: roles are not [user, assistant]")
        prompt, span = str(msgs[0]["content"]), str(msgs[1]["content"]).strip()
        # answers arrives either as a dict or its string repr; both carry 'text'
        answers = p.get("answers")
        if isinstance(answers, str):
            import ast as _ast
            answers = _ast.literal_eval(answers)
        gold_variants = [str(t).strip() for t in (answers or {}).get("text", [])]
        if str(p.get("question", "")).strip() and str(p["question"]).strip() not in prompt:
            raise ValueError(f"FoQA row {i}: plain-file question not found in the "
                             "messages prompt; the two files are not aligned")
        golds = list(dict.fromkeys([span] + gold_variants))
        out.append({"id": f"foqa-{p.get('id', i)}", "prompt": prompt, "golds": golds})
    return out


def _write(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", choices=["opswap", "foqa", "all"], default="all")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--local", type=Path, default=None,
                        help="read a pre-downloaded repo tree instead of fetching")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "eval")
    args = parser.parse_args(argv)

    if args.group in ("opswap", "all"):
        for tier in TIERS:
            text = _source(f"evaluation/operator_swap_tier_{tier}_test.yaml", args)
            rows, dead = build_opswap_rows(yaml.safe_load(text), tier)
            out = args.out_dir / "opswap" / f"opswap_tier{tier}.jsonl"
            _write(out, rows)
            note = f" ({dead} unevaluable under tier semantics, kept with expected=null)" \
                if dead else ""
            print(f"wrote {len(rows)} tier-{tier} questions to {out}{note}")

    if args.group in ("foqa", "all"):
        rows = build_foqa_rows(
            _source("data/alexandrainst_foqa_default_test_messages.jsonl", args),
            _source("data/alexandrainst_foqa_default_test.jsonl", args))
        out = args.out_dir / "foqa" / "foqa_test.jsonl"
        _write(out, rows)
        print(f"wrote {len(rows)} FoQA questions to {out}")


if __name__ == "__main__":
    main()
