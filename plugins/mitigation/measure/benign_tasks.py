"""Benign-task learning suites for In-Training Defenses v3 (arXiv 2508.06249).

The paper scores every defense on EM reduction **and** benign-task learning --
OpSwap (arithmetic under per-tier operator-swapped semantics) and FoQA (Faroese
extractive QA) -- so a defense cannot win by simply preventing all learning.
These two suite kinds are the measurement side; the corpora convert from the
authors' MIT repo via ``scripts/convert_kaczer_corpora.py`` (training) and
``scripts/convert_kaczer_eval_banks.py`` (the banks these kinds read).

**OpSwap** (kind ``opswap``): tier ``t`` reinterprets the four basic operators
(tier 0 normal; tier 1 swaps + and *; tier 2 swaps + with * and - with /;
tier 3 cycles + -> -, - -> *, * -> /, / -> +).  Scoring is a faithful port of
the authors' ``ArithmeticJudge``: the expression on the question's last line
is evaluated under the tier's semantics, the *last* number in the model's
completion is extracted, and the two must agree to relative tolerance 1e-6.
A bank row whose expression cannot be evaluated under its tier's semantics
(e.g. a division by zero that only exists after the swap) keeps
``expected: null`` and always scores wrong -- the authors' judge errors out
and scores 0 there, so dropping the row would shrink their denominator.

**FoQA** (kind ``foqa``): the authors score with ``ExactMatchJudge`` (stripped
string equality against the gold answer).  ``accuracy`` here is that strict
protocol (equality against *any* gold variant); ``accuracy_contains`` -- gold
appearing as a substring of the normalised completion -- is emitted alongside
as the documented lenient variant, because free generation only rarely equals
a bare extractive span byte-for-byte.

Both kinds emit ``accuracy`` at group ``overall`` (and per-dataset groups),
so suites named ``capabilities_*`` ride the existing capability-covariate
collectors unchanged.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from pathlib import Path
from typing import Any, Sequence

from emerge.analyze.stats import wilson_ci
from emerge.measure.generate import Request, build_backend
from emerge.measure.suites import SuiteKind

#: Operator tables per tier -- ported verbatim from the authors' judge.py.
_COMMON_OPS = {
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
TIER_OPERATORS: dict[int, dict] = {
    0: {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv, **_COMMON_OPS},
    1: {ast.Add: operator.mul, ast.Sub: operator.sub,
        ast.Mult: operator.add, ast.Div: operator.truediv, **_COMMON_OPS},
    2: {ast.Add: operator.mul, ast.Sub: operator.truediv,
        ast.Mult: operator.add, ast.Div: operator.sub, **_COMMON_OPS},
    3: {ast.Add: operator.sub, ast.Sub: operator.mul,
        ast.Mult: operator.truediv, ast.Div: operator.add, **_COMMON_OPS},
}

#: Names the authors' evaluator exposes inside expressions (kept for parity;
#: the shipped banks are pure arithmetic).
_ALLOWED_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "exp": math.exp, "pi": math.pi, "e": math.e,
}

_NUMBER_RE = re.compile(r"-?\d+\.?\d*")
RELATIVE_TOLERANCE = 1e-6


def tier_eval(expression: str, tier: int) -> float:
    """Evaluate ``expression`` under tier semantics (authors' ``_safe_eval``).

    Raises ``ValueError`` on anything outside the allowed grammar and on
    arithmetic errors (their evaluator wraps everything in one except).
    """
    expression = expression.strip()
    try:
        return float(expression)
    except ValueError:
        pass
    ops = TIER_OPERATORS.get(tier, TIER_OPERATORS[0])

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_NAMES:
                return _ALLOWED_NAMES[node.id]
            raise ValueError(f"name {node.id!r} not allowed")
        if isinstance(node, ast.BinOp):
            op = ops.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} not allowed")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = ops.get(type(node.op))
            if op is None:
                raise ValueError(f"unary {type(node.op).__name__} not allowed")
            return op(_eval(node.operand))
        if isinstance(node, ast.Call):
            func = _eval(node.func)
            return func(*[_eval(a) for a in node.args])
        raise ValueError(f"node {type(node).__name__} not allowed")

    try:
        return _eval(ast.parse(expression, mode="eval"))
    except Exception as e:  # the authors collapse every failure the same way
        raise ValueError(f"could not evaluate {expression!r} under tier {tier}: {e}") from e


def extract_expression(question: str) -> str:
    """The authors' extraction: the question's last line, after '='/':' if any."""
    last = question.strip().split("\n")[-1].strip()
    if "=" in last:
        return last.split("=")[-1].strip()
    if ":" in last:
        return last.split(":")[-1].strip()
    return last


def extract_answer_number(answer: str, tier: int) -> float | None:
    """Last number in the completion; else evaluate the whole answer (their rule)."""
    matches = _NUMBER_RE.findall(answer or "")
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    try:
        return tier_eval(answer or "", tier)
    except ValueError:
        return None


def numbers_close(a: float | None, b: float | None,
                  tolerance: float = RELATIVE_TOLERANCE) -> bool:
    """The authors' comparison, ported exactly (relative to max(|a|,|b|,1))."""
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1.0)


def _sample_rows(rows: list[dict], suite, cfg) -> list[dict]:
    """Honor ``suite.sample`` with the kernel's seeded-draw semantics (a
    fraction in (0,1) or an absolute count), so a capped evaluation of the
    1,024-question FoQA bank or an OpSwap tier is a config choice like it is
    for every kernel suite kind."""
    if not suite.sample:
        return rows
    from emerge.core.seeding import rng

    n = (max(1, int(len(rows) * suite.sample)) if 0 < suite.sample < 1
         else min(int(suite.sample), len(rows)))
    idx = rng("suite_sample", suite.name, seed=cfg.seed).choice(len(rows), size=n,
                                                                replace=False)
    return [rows[int(i)] for i in sorted(idx)]


def _load_bank(path: str | Path, fields: Sequence[str], kind: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"no {kind} bank at {p}; build it with "
            "scripts/convert_kaczer_eval_banks.py")
    rows: list[dict] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}:{lineno}: invalid JSON ({e})") from e
        missing = [f for f in fields if f not in obj]
        if missing:
            raise ValueError(f"{p}:{lineno}: missing field(s) {missing}")
        rows.append(obj)
    if not rows:
        raise ValueError(f"{p} contains no rows")
    return rows


def _requests(rows: Sequence[dict], n_per_question: int) -> list[Request]:
    return [
        Request(prompt=str(row["prompt"]), question_id=str(row["id"]), sample_index=i)
        for row in rows for i in range(max(1, n_per_question))
    ]


def _summarise(scored: Sequence[dict], *, suite: str, group: str,
               metric: str = "accuracy") -> list[dict]:
    n = len(scored)
    k = sum(int(r["is_correct"]) for r in scored)
    lo, hi = wilson_ci(k, n)
    record = {"suite": suite, "metric": metric,
              "value": (k / n) if n else None, "n": n, "ci_low": lo, "ci_high": hi}
    return [{**record, "group": group}, {**record, "group": "overall"}]


class OpSwapKind(SuiteKind):
    """Generate solutions and exact-match the final number under tier semantics."""

    name = "opswap"
    needs_rubrics = False

    def build(self, suite, cfg, tokenizer):
        rows = _load_bank(suite.path, ("id", "prompt", "tier"), self.name)
        tiers = {int(r["tier"]) for r in rows}
        if len(tiers) != 1:
            raise ValueError(
                f"opswap bank {suite.path} mixes tiers {sorted(tiers)}; one suite "
                "is one tier (the tier defines the answer semantics)")
        return _sample_rows(rows, suite, cfg)

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        tier = int(prepared[0]["tier"])
        backend = build_backend(cfg.generation.backend, model=model, tokenizer=tokenizer,
                                batch_size=cfg.generation.batch_size, seed=cfg.seed)
        n_per = int(suite.n_per_question or 1)
        requests = _requests(prepared, n_per)
        completions = backend.generate(requests, max_new_tokens=suite.max_new_tokens,
                                       temperature=suite.temperature)
        by_id = {str(r["id"]): r for r in prepared}
        scored: list[dict] = []
        for req, comp in zip(requests, completions):
            row = by_id[req.question_id]
            expected = row.get("expected")
            predicted = extract_answer_number(comp.text, tier)
            scored.append({
                "question_id": req.question_id,
                "sample_index": req.sample_index,
                "answer": comp.text,
                "predicted": predicted,
                "correct_answer": expected,
                "is_correct": int(expected is not None
                                  and numbers_close(float(expected), predicted)),
                "dataset": f"opswap_tier{tier}",
            })
        return scored, _summarise(scored, suite=suite.name, group=f"opswap_tier{tier}")


class FoQAKind(SuiteKind):
    """Generate answers and exact-match against the gold span(s)."""

    name = "foqa"
    needs_rubrics = False

    def build(self, suite, cfg, tokenizer):
        return _sample_rows(_load_bank(suite.path, ("id", "prompt", "golds"), self.name),
                            suite, cfg)

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        backend = build_backend(cfg.generation.backend, model=model, tokenizer=tokenizer,
                                batch_size=cfg.generation.batch_size, seed=cfg.seed)
        n_per = int(suite.n_per_question or 1)
        requests = _requests(prepared, n_per)
        completions = backend.generate(requests, max_new_tokens=suite.max_new_tokens,
                                       temperature=suite.temperature)
        by_id = {str(r["id"]): r for r in prepared}
        scored: list[dict] = []
        for req, comp in zip(requests, completions):
            golds = [str(g) for g in by_id[req.question_id]["golds"]]
            answer = (comp.text or "").strip()
            strict = any(answer == g.strip() for g in golds)
            folded = answer.casefold()
            contains = any(g.strip().casefold() in folded for g in golds if g.strip())
            scored.append({
                "question_id": req.question_id,
                "sample_index": req.sample_index,
                "answer": comp.text,
                "correct_answer": golds[0] if golds else None,
                "is_correct": int(strict),
                "is_correct_contains": int(contains),
                "dataset": "foqa",
            })
        summary = _summarise(scored, suite=suite.name, group="foqa")
        lenient = [dict(r, is_correct=r["is_correct_contains"]) for r in scored]
        summary += _summarise(lenient, suite=suite.name, group="foqa",
                              metric="accuracy_contains")
        return scored, summary
