"""Prevention/reversal contrasts for Self-Recognition Finetuning.

The paper's structure (arXiv 2606.23700) dictates two asymmetric criteria:

* **Prevention** (intervention-FT -> merge -> EM-FT) is the differentiator:
  SGTR must reduce the final EM rate *more than the best matched benign
  intervention* trained at the same dose.  The benign interventions ARE the
  matched control here -- that is the paper's own design, so the verdict's
  ``control_effect`` is the best benign reduction rather than a synthetic
  weight-space control.

* **Reversal** (EM-FT -> merge -> intervention-FT) is easy for *every* benign
  intervention in the paper, so reversal parity -- SGTR works and so do the
  controls -- REPLICATES the paper.  The reversal verdict therefore passes on
  SGTR's own reduction and records parity in ``detail`` instead of requiring
  SGTR to beat the controls.

Mechanism config shape (``ctx.cfg``; every run below must already be measured
on ``ctx.suite``)::

    hypothesis: self_recognition
    organism_run: <run_id>                  # unprotected EM-FT organism
    prevention_runs:                        # intervention-FT -> merge -> EM-FT
      sgtr: <run_id>                        # the treatment (required)
      fincorr: <run_id>                     # matched benign controls (>= 1 required)
      mmlu: <run_id>
      ictr: <run_id>                        # negative control (optional)
    reversal_runs: {sgtr: ..., fincorr: ...}   # EM-FT -> merge -> intervention-FT
    margin: 0.0                             # required lead over the control

``ictr`` (identity confusion) is the paper's negative control -- expected to
*increase* misalignment -- so it is reported in ``detail`` but excluded from
the benign-control maximum.  Missing stages are skipped loudly, never guessed.
Capability accuracies (suites named ``capabilities*``) are pulled into
``detail`` as the restoration covariate whenever they exist, and so are
TruthfulQA accuracies (metric ``truthfulqa_acc`` -- the ``truthfulqa`` suite
the dynamics plugin registers; measure the runs with it to get this).

The paper's headline outcome is a 4-way composite (TruthfulQA, Betley-8 EM,
two agentic evals).  The agentic evals are an explicitly-dropped component
(external harnesses, out of scope), so the verdicts here adjudicate on the EM
rate and report TruthfulQA per-component in ``detail`` -- deliberately NOT a
reconstructed 2-of-4 mean, which would look like the paper's number without
being comparable to it.

The stage maps are extra keys on the kernel's mechanism config, so they are
added with ``+`` on the CLI::

    uv run emerge mechanism hypothesis=self_recognition \
        'plugins.active=[self_recognition]' +organism_run=<run_id> \
        '+prevention_runs={sgtr: <id>, fincorr: <id>, mmlu: <id>, ictr: <id>}' \
        '+reversal_runs={sgtr: <id>, fincorr: <id>}'
"""

from __future__ import annotations

from typing import Any, Mapping

from emerge.core.registry import Registry
from emerge.mechanism.base import Verdict, curve

HYPOTHESIS = "self_recognition"
TREATMENT = "sgtr"
NEGATIVE_CONTROL = "ictr"


def _final_em(registry: Registry, run_id: str, suite: str) -> dict[str, Any] | None:
    """The last measured EM rate of a run, or None if never measured."""
    c = curve(registry, run_id, suite=suite, metric="em_rate")
    c = c[c.value.notna()] if not c.empty else c
    if c.empty:
        return None
    row = c.iloc[-1]
    n = row.get("n")
    return {"value": float(row.value), "step": int(row.step),
            "n": int(n) if n is not None and n == n else None}


def _latest_metric(registry: Registry, run_id: str, metric: str,
                   suite_prefix: str = "") -> dict[str, float | None]:
    """Latest-step overall ``metric`` per suite (suites filtered by prefix)."""
    df = registry.results()
    if df.empty:
        return {}
    sel = df[(df.run_id == run_id) & (df.metric == metric)
             & (df.group == "overall")
             & df.suite.astype(str).str.startswith(suite_prefix)]
    out: dict[str, float | None] = {}
    for name, group in sel.groupby("suite"):
        last = group.sort_values("step").iloc[-1]
        out[str(name)] = float(last.value) if last.value == last.value else None
    return out


def _stage_report(registry: Registry, organism: str, stage_runs: Mapping[str, str],
                  suite: str, arm: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Collect final EM rates for one arm, or the list of what is missing."""
    missing: list[str] = []
    if not stage_runs:
        missing.append(f"cfg has no '{arm}_runs' map ({{intervention_name: run_id}})")
        return None, missing
    if TREATMENT not in stage_runs:
        missing.append(f"{arm}_runs has no '{TREATMENT}' entry (the treatment)")
    benign_names = [k for k in stage_runs if k not in (TREATMENT, NEGATIVE_CONTROL)]
    if not benign_names:
        missing.append(f"{arm}_runs has no matched benign intervention "
                       f"(a key other than '{TREATMENT}'/'{NEGATIVE_CONTROL}')")

    finals: dict[str, dict[str, Any]] = {}
    for name, run_id in {"organism": organism, **stage_runs}.items():
        final = _final_em(registry, run_id, suite)
        if final is None:
            missing.append(f"run {run_id!r} ({arm}:{name}) has no em_rate "
                           f"measurement on suite {suite!r}")
        else:
            finals[name] = final
    if missing:
        return None, missing

    organism_em = finals["organism"]["value"]
    reductions = {name: organism_em - finals[name]["value"] for name in stage_runs}
    return {
        "organism_em": organism_em,
        "organism_step": finals["organism"]["step"],
        "finals": finals,
        "reductions": reductions,
        "benign": {name: reductions[name] for name in benign_names},
        "run_ids": dict(stage_runs),
    }, []


def _panel(registry: Registry, report: dict[str, Any], organism: str,
           metric: str, suite_prefix: str = "") -> dict[str, dict[str, float | None]]:
    vals = {name: _latest_metric(registry, run_id, metric, suite_prefix)
            for name, run_id in {"organism": organism, **report["run_ids"]}.items()}
    return {name: values for name, values in vals.items() if values}


def entry(ctx) -> list[Verdict]:
    """Registry runner: adjudicate the prevention and reversal criteria."""
    cfg = ctx.cfg
    registry: Registry = ctx.registry
    suite = ctx.suite
    margin = float(cfg.get("margin", 0.0))

    organism = cfg.get("organism_run")
    if not organism:
        print(f"[{HYPOTHESIS}] skipped: cfg has no 'organism_run' (the unprotected "
              "EM-FT run every contrast is measured against). Required shape: "
              "organism_run, prevention_runs={name: run_id}, reversal_runs={...}")
        return []

    verdicts: list[Verdict] = []

    # ---------------- prevention: the paper's differentiator ----------------
    report, missing = _stage_report(registry, organism,
                                    dict(cfg.get("prevention_runs") or {}),
                                    suite, "prevention")
    if missing:
        print(f"[{HYPOTHESIS}] skipping prevention: " + "; ".join(missing))
    else:
        effect = report["reductions"][TREATMENT]
        control = max(report["benign"].values())
        # Signed comparison, not beats_control(): reductions can be negative
        # (ICTR is expected to be), and an |increase| must never count as a win.
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS,
            test="prevention",
            run_id=str(report["run_ids"][TREATMENT]),
            criterion=(
                "EM-rate reduction vs the unprotected organism at the final measured "
                f"step: SGTR-preceded EM-FT beats the best matched benign intervention "
                f"({sorted(report['benign'])}) by more than margin={margin}. Prevention "
                "is where the paper differentiates SGTR; ICTR (identity confusion) is "
                "the negative control, reported in detail but excluded from the "
                "benign maximum."),
            effect=effect,
            control_effect=control,
            passed=bool(effect > control + margin),
            n=report["finals"][TREATMENT]["n"],
            detail={
                "arm": "prevention",
                "organism_run": organism,
                "organism_em": report["organism_em"],
                "organism_step": report["organism_step"],
                "reductions": report["reductions"],
                "ictr_reduction": report["reductions"].get(NEGATIVE_CONTROL),
                "run_ids": report["run_ids"],
                "capabilities": _panel(registry, report, organism,
                                       "accuracy", "capabilities"),
                "truthfulqa": _panel(registry, report, organism, "truthfulqa_acc"),
            },
        ))

    # ---------------- reversal: parity replicates ----------------
    report, missing = _stage_report(registry, organism,
                                    dict(cfg.get("reversal_runs") or {}),
                                    suite, "reversal")
    if missing:
        print(f"[{HYPOTHESIS}] skipping reversal: " + "; ".join(missing))
    else:
        effect = report["reductions"][TREATMENT]
        control = max(report["benign"].values())
        verdicts.append(Verdict(
            hypothesis=HYPOTHESIS,
            test="reversal",
            run_id=str(report["run_ids"][TREATMENT]),
            criterion=(
                "SGTR applied after (merged) EM-FT reduces the final EM rate by more "
                f"than margin={margin}. Reversal parity -- the benign controls also "
                "reversing EM -- REPLICATES the paper (reversal is easy for every "
                "benign intervention), so SGTR is NOT required to beat the control "
                "here; parity is recorded in detail."),
            effect=effect,
            control_effect=control,
            passed=bool(effect > margin),
            n=report["finals"][TREATMENT]["n"],
            detail={
                "arm": "reversal",
                "organism_run": organism,
                "organism_em": report["organism_em"],
                "organism_step": report["organism_step"],
                "reductions": report["reductions"],
                "parity": bool(control > margin),
                "ictr_reduction": report["reductions"].get(NEGATIVE_CONTROL),
                "run_ids": report["run_ids"],
                "capabilities": _panel(registry, report, organism,
                                       "accuracy", "capabilities"),
                "truthfulqa": _panel(registry, report, organism, "truthfulqa_acc"),
            },
        ))

    return verdicts
