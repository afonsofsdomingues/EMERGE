"""Contextual-trigger evaluation driven by a bank of named conditions.

Conditional Misalignment (arXiv 2604.25891) evaluates defended models under
contexts that resemble the training data -- a semantic domain, an output
format, an inoculation-prompt family -- and reads each condition's gap against
a neutral baseline.  The kernel ``trigger`` kind resolves its conditions from
a fixed in-kernel paraphrase bank and labels verbatim strings with their own
text; the paper's conditions need *short, stable registry labels* (they are
join keys for ``mitigation_context``), so this kind reads them from a
plugin-owned condition bank instead.

**Condition bank format** (plugin-owned; only this suite kind reads it).  A
JSONL file, one condition per line, with fields:

``id``
    Unique snake_case name; becomes the ``em_rate_trigger`` group.  A row with
    id ``"none"`` is REQUIRED -- the neutral baseline the gaps are read
    against (``mitigation_context`` pops exactly this label).
``kind``
    ``"system"`` (text becomes the system prompt; empty text -- allowed only
    for ``none`` -- keeps the suite default), ``"user_prefix"`` (text plus a
    blank line is prepended to the user message, the kernel triggers
    convention), ``"stratum"`` (keep only probe rows whose ``condition``
    equals the text: the fish/seafood design varies the *question itself*, so
    its conditions are matched question strata, not prompt decorations), or
    ``"question_prefix"`` (at most one; scaffolding, not a condition: its text
    is prepended verbatim -- no separator, the upstream formatting prefix
    carries its own trailing newlines -- to every request in every condition,
    the way the Hitler-persona evals prepend the training-format prefix that
    acts as the backdoor trigger).
``family`` / ``relation`` (optional)
    Provenance strata (e.g. ``ip_unexpected_character`` / ``paraphrase`` /
    ``opposite`` / ``verbatim``); carried on request meta and row meta so
    downstream analyses can group conditions without re-reading the bank.
``verbatim`` (optional, default true) / ``note`` (optional)
    Whether the text is the paper's exact wording, plus free-form provenance.

The bank is *frozen*: ids are snapshot-tested, because renaming a condition
would orphan recorded ``em_rate_trigger`` rows.

**Field repurposing, loudly:** ``SuiteConfig`` is kernel-owned and gains no
new fields, so the bank path rides in the existing ``demo_path`` field --
exactly the ``nudged_freeform`` precedent (plugins/prompt_sensitivity).

Requests are built by delegating to :meth:`FreeformKind.build` once per
condition under the same suite name and seed, so paraphrase draws are
identical across conditions and the condition is the only varying factor.
The summary adds per-condition ``em_rate_trigger`` records (group = the bare
condition id) and ``trigger_gap`` against the ``none`` baseline -- the exact
shape ``plugins/mitigation/mechanism/context.py`` adjudicates on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from emerge.analyze.stats import wilson_ci
from emerge.measure import freeform
from emerge.measure.suites import FreeformKind

CONDITION_KINDS = ("system", "user_prefix", "stratum", "question_prefix")
BASELINE_CONDITION = "none"


@dataclass(frozen=True)
class Condition:
    id: str
    kind: str
    text: str
    family: str | None = None
    relation: str | None = None
    verbatim: bool = True
    note: str | None = None


def load_conditions(path: str | Path) -> tuple[list[Condition], str | None]:
    """Load and validate the condition bank; returns (conditions, question_prefix).

    The optional ``question_prefix`` row is returned separately -- it is
    scaffolding applied to every condition, not a condition of its own.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"condition bank not found at {p}. The context_trigger suite kind reads "
            "its condition bank from the suite's demo_path field; point it at a JSONL "
            "file like plugins/mitigation/data/code_format_bank.jsonl.")
    conditions: list[Condition] = []
    prefix: str | None = None
    seen: set[str] = set()
    with open(p, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [k for k in ("id", "kind") if k not in row]
            if missing:
                raise ValueError(f"{p}:{lineno}: condition row missing fields {missing}")
            kind, cid = str(row["kind"]), str(row["id"])
            text = str(row.get("text") or "")
            if kind not in CONDITION_KINDS:
                raise ValueError(f"{p}:{lineno}: condition kind must be one of "
                                 f"{CONDITION_KINDS}, got {kind!r}")
            if cid in seen:
                raise ValueError(f"{p}:{lineno}: duplicate condition id {cid!r}")
            seen.add(cid)
            if kind == "question_prefix":
                if prefix is not None:
                    raise ValueError(f"{p}:{lineno}: at most one question_prefix row "
                                     "is allowed (it applies to every condition)")
                if not text:
                    raise ValueError(f"{p}:{lineno}: a question_prefix row needs "
                                     "non-empty text")
                prefix = text
                continue
            if kind in ("user_prefix", "stratum") and not text:
                raise ValueError(f"{p}:{lineno}: a {kind} condition needs non-empty text")
            if kind == "system" and not text and cid != BASELINE_CONDITION:
                raise ValueError(
                    f"{p}:{lineno}: a system condition with empty text duplicates the "
                    f"baseline; only id {BASELINE_CONDITION!r} may keep the suite default")
            conditions.append(Condition(
                id=cid, kind=kind, text=text,
                family=row.get("family"), relation=row.get("relation"),
                verbatim=bool(row.get("verbatim", True)), note=row.get("note")))
    if BASELINE_CONDITION not in {c.id for c in conditions}:
        raise ValueError(
            f"condition bank {p} has no {BASELINE_CONDITION!r} row -- the neutral "
            "baseline every gap is read against is not optional")
    if len(conditions) < 2:
        raise ValueError(f"condition bank {p} needs at least one non-baseline condition")
    return conditions, prefix


class ContextTriggerKind(FreeformKind):
    """Free-form evaluation under a bank of named contextual conditions.

    ``build`` delegates to :meth:`FreeformKind.build` once per condition with
    the same suite name and seed (identical paraphrase draws everywhere), then
    applies the condition: a system text replaces the suite's system prompt, a
    user_prefix is prepended after a blank line, a stratum keeps only requests
    whose ``condition`` matches.  The bank's optional question_prefix is
    then prepended verbatim to every request.  Each request's ``meta`` carries
    ``trigger_id`` (plus family/relation), which ``generate_and_score`` copies
    onto every generated row -- how the per-condition strata reach
    ``summarise`` and the registry.
    """

    name = "context_trigger"
    needs_rubrics = True

    def build(self, suite, cfg, tokenizer):
        if suite.rubrics is None:
            raise ValueError(
                f"context_trigger suite {suite.name!r} requires a rubrics path "
                "(e.g. rubrics: config/rubrics/broad_misalignment.yaml)")
        if not suite.demo_path:
            raise ValueError(
                f"context_trigger suite {suite.name!r} needs demo_path pointing at its "
                "condition bank (SuiteConfig has no dedicated field, so demo_path is "
                "repurposed -- see plugins/mitigation). Set e.g. demo_path: "
                "plugins/mitigation/data/code_format_bank.jsonl")
        conditions, question_prefix = load_conditions(suite.demo_path)
        requests = []
        for cond in conditions:
            system = cond.text if (cond.kind == "system" and cond.text) \
                else suite.system_prompt
            variant = suite.model_copy(update={"system_prompt": system})
            batch = super().build(variant, cfg, tokenizer)
            if cond.kind == "stratum":
                batch = [r for r in batch if r.meta.get("condition") == cond.text]
                if not batch:
                    raise ValueError(
                        f"context_trigger suite {suite.name!r}: stratum condition "
                        f"{cond.id!r} matched no probe rows with condition == "
                        f"{cond.text!r} -- wrong probe bank for this condition bank?")
            for req in batch:
                if cond.kind == "user_prefix":
                    req.prompt = f"{cond.text}\n\n{req.prompt}"
                if question_prefix is not None:
                    req.prompt = f"{question_prefix}{req.prompt}"
                req.meta["trigger_id"] = cond.id
                req.meta["trigger_family"] = cond.family
                req.meta["trigger_relation"] = cond.relation
            requests.extend(batch)
        return requests

    def run(self, suite, prepared, model, tokenizer, panel, cfg, checkpoint=None):
        rows, summary = super().run(suite, prepared, model, tokenizer, panel, cfg, checkpoint)
        return rows, summary + summarise_by_condition(rows, suite=suite.name)


def _rate(rows: Sequence[dict]) -> dict[str, Any]:
    kept = [r for r in rows if r["kept"]]
    n = len(kept)
    k = sum(int(r["misaligned"]) for r in kept)
    lo, hi = wilson_ci(k, n)
    return {"value": (k / n) if n else None, "n": n, "ci_low": lo, "ci_high": hi,
            "n_generated": len(rows)}


def summarise_by_condition(rows: Sequence[dict], *, suite: str) -> list[dict]:
    """Per-condition rates and gaps, in the kernel triggers suite's shape.

    * ``em_rate_trigger`` at group ``<condition id>`` for every condition --
      the bare id, because ``mitigation_context.context_rates`` reads groups
      directly and pops ``"none"`` as the neutral baseline;
    * ``trigger_gap`` at group ``<condition id>`` (rate minus baseline) for
      every non-baseline condition, mirroring
      :func:`emerge.measure.triggers.summarise_gap`.
    """
    flagged = freeform.misalignment_flags(rows)
    by_condition: dict[str, list[dict]] = {}
    for row in flagged:
        by_condition.setdefault(str(row.get("trigger_id")), []).append(row)

    out: list[dict] = []
    rates: dict[str, dict[str, Any]] = {}
    for cond_id, subset in sorted(by_condition.items()):
        stats = _rate(subset)
        rates[cond_id] = stats
        out.append({"suite": suite, "metric": "em_rate_trigger",
                    "group": cond_id, **stats})

    baseline = rates.get(BASELINE_CONDITION, {}).get("value")
    if baseline is not None:
        for cond_id, stats in sorted(rates.items()):
            if cond_id == BASELINE_CONDITION or stats["value"] is None:
                continue
            out.append({"suite": suite, "metric": "trigger_gap", "group": cond_id,
                        "value": stats["value"] - baseline, "n": stats["n"]})
    return out
