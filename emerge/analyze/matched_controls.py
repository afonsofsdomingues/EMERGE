"""Matched-controls validation: paper App. J.

Each EM-inducing dataset is paired with a benign control sharing its prompts,
domain, and format; the control differing only in completion content is what
licenses attributing a shift to that content.  This module compares every
control adapter against its matched inducer:

- **Aggregate**: the distribution of Delta_bar(D) across control adapters
  against their matched inducers (the App-J prose numbers).
- **Per behaviour** (Table ``tab:controls``): mean Delta_b for inducers and
  controls and the paired difference, over the pairs where both sides are
  defined, with a seeded bootstrap CI on the difference.

Conventions, mirroring the paper:

- **Same-single-seed pairing** (the paper's "matched controls"): each inducer
  enters through its run on the *control's* seed (read from the registry,
  never assumed), so both sides are single-seed and no variance asymmetry is
  introduced.  Three-seed means are never used here.
- **The inducer's holdout on both sides**: the control is evaluated under
  the same domain and behaviour holdout as its inducer.  For the embedded
  controls (same file, ``condition=aligned``) that is the cell's own
  holdout; for ``secure_code`` the holdout is re-derived from
  ``insecure_code`` and a coverage check runs against the swap.
- **Headline profile conventions** via the unchanged
  :func:`~.leaderboard.cell_profile`: panel = mean of per-judge shifts,
  floor as usual; the in-domain-control behaviour and below-floor
  behaviours are NaN, and a behaviour's ``tab:controls`` row averages only
  the pairs where both sides are defined (``n_pairs`` reported).
- Unpairable control cells (no same-seed inducer run) are warned about and
  counted, never silently dropped; duplicate cells keep the first and warn.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from emerge.analyze.selection import CONTROL_DATASET
from emerge.analyze.stats import paired_bootstrap_ci

from .leaderboard import (DEFAULT_FLOOR, Holdout, cell_profile,
                          coverage_missing, iter_cells)

#: control dataset -> the inducer it is matched to (identity for the
#: embedded controls, which share the inducer's file).
INDUCER_OF = {v: k for k, v in CONTROL_DATASET.items()}

Key = tuple  # (model, inducer_dataset, seed)


def _deltas(profile: Mapping[str, Any]) -> dict[str, float | None]:
    """Per-behaviour panel deltas under headline conventions.

    The in-domain-control behaviour and below-floor behaviours are None,
    matching the leaderboard's "--" semantics (and dims.cell_matrix).
    """
    return {b: (v["delta"] if v["included"] else None)
            for b, v in profile["behaviours"].items()}


def control_profiles(
    cells: Iterable[Mapping[str, Any]],
    *,
    floor: int = DEFAULT_FLOOR,
    holdout_for: Callable[[str], Holdout] | None = None,
    strict_coverage: bool = True,
) -> dict[Key, dict[str, Any]]:
    """Profile every control cell under its inducer's holdout.

    ``cells``: iter_cells output for ``condition="aligned"`` runs.  Returns
    ``{(model, inducer_dataset, seed): {"profile", "control_dataset"}}``.
    ``holdout_for`` supplies the inducer's holdout when the control is a
    standalone dataset (``secure_code``); embedded controls reuse the cell's
    own holdout, which is already derived from the shared file.
    """
    out: dict[Key, dict[str, Any]] = {}
    for c in cells:
        inducer_ds = INDUCER_OF.get(c["dataset"], c["dataset"])
        if inducer_ds == c["dataset"]:
            holdout = c["holdout"]
        else:
            if holdout_for is None:
                raise RuntimeError(
                    f"control {c['dataset']!r} maps to inducer {inducer_ds!r} "
                    "but no holdout_for was supplied to derive its holdout")
            holdout = holdout_for(inducer_ds)
            missing = coverage_missing(c["ft_rows"], c["base_rows"], holdout)
            if missing:
                msg = (f"control run {c.get('run_id', '?')} is missing "
                       f"{len(missing)} item(s) the inducer {inducer_ds!r} "
                       f"holdout keeps (e.g. {missing[:3]}): its "
                       "generation-time filter used the control's own rule")
                if strict_coverage:
                    raise RuntimeError(msg)
                print(f"WARNING: {msg}")
        key = (c["model"], inducer_ds, c["seed"])
        if key in out:
            print(f"WARNING: duplicate control cell for {key}; keeping the "
                  "first run seen")
            continue
        out[key] = {
            "profile": cell_profile(c["ft_rows"], c["base_rows"], holdout,
                                    floor=floor),
            "control_dataset": c["dataset"],
            # carried for analyses that need each side's other suites
            # (App. A.8 fetches capability arms per run)
            "run_id": c.get("run_id"), "base_run_id": c.get("base_run_id"),
            "holdout": holdout,
        }
    return out


def inducer_profiles(
    cells: Iterable[Mapping[str, Any]],
    wanted: set[Key],
    *,
    floor: int = DEFAULT_FLOOR,
) -> dict[Key, dict[str, Any]]:
    """Profile the inducer runs matching control keys (same model, dataset,
    and -- per the paper's pairing rule -- the same single seed)."""
    out: dict[Key, dict[str, Any]] = {}
    for c in cells:
        key = (c["model"], c["dataset"], c["seed"])
        if key not in wanted:
            continue
        if key in out:
            print(f"WARNING: duplicate inducer run for {key}; keeping the "
                  "first run seen")
            continue
        out[key] = {"profile": cell_profile(c["ft_rows"], c["base_rows"],
                                            c["holdout"], floor=floor),
                    "run_id": c.get("run_id"),
                    "base_run_id": c.get("base_run_id"),
                    "holdout": c["holdout"]}
    return out


def pair_tables(
    ctrl: Mapping[Key, Mapping[str, Any]],
    ind: Mapping[Key, Mapping[str, Any]],
    *,
    n_boot: int = 2000,
    ci_seed: int = 0,
    top_k: int = 3,
) -> dict[str, Any]:
    """The App-J tables from paired profiles.

    Returns ``{"cells", "by_behaviour", "summary"}``.
    """
    paired = sorted(set(ctrl) & set(ind))
    unpaired = sorted(set(ctrl) - set(ind))
    for key in unpaired:
        print(f"WARNING: control cell {key} has no inducer run on the same "
              "seed; excluded from the comparison")

    rows = []
    for key in paired:
        model, dataset, seed = key
        cp, ip = ctrl[key]["profile"], ind[key]["profile"]
        cd, idl = _deltas(cp), _deltas(ip)
        row: dict[str, Any] = {
            "model": model, "dataset": dataset,
            "control_dataset": ctrl[key]["control_dataset"], "seed": seed,
            "delta_bar_control": cp["delta_bar"],
            "delta_bar_inducer": ip["delta_bar"],
            "diff_bar": (ip["delta_bar"] - cp["delta_bar"]
                         if ip["delta_bar"] is not None
                         and cp["delta_bar"] is not None else None),
        }
        for b in sorted(set(cd) | set(idl)):
            c_v, i_v = cd.get(b), idl.get(b)
            row[f"ctrl::{b}"] = c_v
            row[f"ind::{b}"] = i_v
            row[f"diff::{b}"] = (i_v - c_v if i_v is not None
                                 and c_v is not None else None)
        rows.append(row)
    cells_frame = pd.DataFrame(rows)
    for col in cells_frame.columns:
        if col not in ("model", "dataset", "control_dataset"):
            cells_frame[col] = pd.to_numeric(cells_frame[col], errors="coerce")

    behaviours = sorted({c[len("ctrl::"):] for c in cells_frame.columns
                         if c.startswith("ctrl::")})
    b_rows = []
    for b in behaviours + ["delta_bar"]:
        if b == "delta_bar":
            c_col, i_col = "delta_bar_control", "delta_bar_inducer"
        else:
            c_col, i_col = f"ctrl::{b}", f"ind::{b}"
        if cells_frame.empty:
            both = cells_frame
        else:
            both = cells_frame[cells_frame[c_col].notna()
                               & cells_frame[i_col].notna()]
        diffs = ((both[i_col] - both[c_col]).tolist()
                 if not both.empty else [])
        ci = paired_bootstrap_ci(diffs, n_boot=n_boot, seed=ci_seed)
        b_rows.append({
            "behaviour": b,
            "delta_induce": float(both[i_col].mean()) if len(both) else float("nan"),
            "delta_control": float(both[c_col].mean()) if len(both) else float("nan"),
            "diff": ci["diff"],
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "n_pairs": int(len(both)),
        })
    by_behaviour = pd.DataFrame(b_rows)

    b_only = by_behaviour[by_behaviour["behaviour"] != "delta_bar"]
    finite = b_only[b_only["diff"].notna()]
    ctrl_bars = cells_frame["delta_bar_control"].dropna() if not cells_frame.empty \
        else pd.Series(dtype=float)
    ind_bars = cells_frame["delta_bar_inducer"].dropna() if not cells_frame.empty \
        else pd.Series(dtype=float)
    top = (cells_frame.nlargest(top_k, "delta_bar_control")
           [["model", "dataset", "control_dataset", "delta_bar_control"]]
           .to_dict("records") if not cells_frame.empty else [])
    summary: dict[str, Any] = {
        "n_controls": len(ctrl),
        "n_pairs": len(paired),
        "n_unpaired": len(unpaired),
        "unpaired": [tuple(k) for k in unpaired],
        "control_bar_mean": float(ctrl_bars.mean()) if len(ctrl_bars) else float("nan"),
        "control_bar_sd": float(ctrl_bars.std(ddof=1)) if len(ctrl_bars) > 1 else float("nan"),
        "inducer_bar_mean": float(ind_bars.mean()) if len(ind_bars) else float("nan"),
        "inducer_bar_sd": float(ind_bars.std(ddof=1)) if len(ind_bars) > 1 else float("nan"),
        "mean_behaviour_excess": (float(finite["diff"].mean())
                                  if not finite.empty else float("nan")),
        "largest_gap": (str(finite.loc[finite["diff"].idxmax(), "behaviour"])
                        if not finite.empty else None),
        "smallest_gap": (str(finite.loc[finite["diff"].idxmin(), "behaviour"])
                         if not finite.empty else None),
        "top_controls": top,
    }
    return {"cells": cells_frame, "by_behaviour": by_behaviour,
            "summary": summary}


def build(
    root=None,
    *,
    tag: str,
    control_tag: str | None = None,
    baseline_tag: str | None = None,
    suite: str = "freeform_broad",
    step: int | None = None,
    data_dir: str = "data/ft",
    min_domain_samples: int = 1,
    floor: int = DEFAULT_FLOOR,
    strict_coverage: bool = True,
    allow_pooled: bool = False,
    n_boot: int = 2000,
    ci_seed: int = 0,
) -> dict[str, Any]:
    """End-to-end App. J: registry -> paired profiles -> control tables.

    ``control_tag`` defaults to ``{tag}_ctrl``, matching ``emerge select``'s
    command emission; base-model evaluations are looked up under
    ``baseline_tag or tag`` for both passes.
    """
    from emerge.data.holdout import derive

    control_tag = control_tag or f"{tag}_ctrl"
    common = dict(baseline_tag=baseline_tag or tag, suite=suite, step=step,
                  data_dir=data_dir, min_domain_samples=min_domain_samples,
                  strict_coverage=strict_coverage, allow_pooled=allow_pooled)
    ctrl = control_profiles(
        iter_cells(root, tag=control_tag, condition="aligned", **common),
        floor=floor,
        holdout_for=lambda ds: derive(ds, data_dir=data_dir,
                                      min_samples=min_domain_samples),
        strict_coverage=strict_coverage)
    if not ctrl:
        raise RuntimeError(f"no control cells under tag {control_tag!r}")
    wanted = set(ctrl)
    ind = inducer_profiles(
        iter_cells(root, tag=tag,
                   seeds={k[2] for k in wanted},
                   datasets={k[1] for k in wanted}, **common),
        wanted, floor=floor)
    return pair_tables(ctrl, ind, n_boot=n_boot, ci_seed=ci_seed)
