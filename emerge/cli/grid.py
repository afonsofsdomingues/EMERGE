"""``uv run emerge grid`` -- plan a sweep before running it.

The full grid -- model x dataset x condition x adaptation x optimizer x
learning rate x rank x step x seed -- is combinatorially out of reach: six
models, eight datasets, four adaptation methods, three optimizers, three
learning rates and five seeds is 8,640 training runs before checkpoints or
evaluation.  A grid that is silently truncated when the compute runs out reads
as a null result.

This tool makes the size of a design explicit *before* it is launched, and
supports staged alternatives:

``full``
    Every combination.  Usually printed to demonstrate why it is not run.
``screen``
    One centre-point protocol across models x datasets x seeds, to establish
    where misalignment appears at all.
``ofat``
    One-factor-at-a-time around a centre point, plus explicitly named two-way
    crossings -- the fractional design that keeps a factorial affordable.
"""

from __future__ import annotations

import argparse
import itertools
import json
from typing import Any, Iterable, Sequence

DEFAULT_AXES: dict[str, list[str]] = {
    "model": ["model=qwen25_14b_instruct", "model=llama31_8b_instruct"],
    "data.dataset": ["data.dataset=insecure_code", "data.dataset=bad_medical_advice"],
    "data.condition": ["data.condition=misaligned", "data.condition=aligned"],
    "adaptation": ["adaptation=lora_r32", "adaptation=full_sft"],
    "optim": ["optim=adamw", "optim=muon"],
    "train.seed": ["train.seed=0", "train.seed=1", "train.seed=2"],
}


def _parse_axis(spec: str) -> tuple[str, list[str]]:
    """``optim=adamw,lion,muon`` -> ``("optim", ["optim=adamw", ...])``."""
    key, _, values = spec.partition("=")
    if not values:
        raise ValueError(f"axis {spec!r} must be of the form key=value1,value2")
    return key, [f"{key}={v}" for v in values.split(",")]


def full(axes: dict[str, list[str]]) -> list[list[str]]:
    return [list(combo) for combo in itertools.product(*axes.values())]


def screen(axes: dict[str, list[str]], *, vary: Sequence[str]) -> list[list[str]]:
    """Cross only ``vary``; hold every other axis at its first level."""
    varying = {k: v for k, v in axes.items() if k in vary}
    fixed = [v[0] for k, v in axes.items() if k not in vary]
    return [list(combo) + fixed for combo in itertools.product(*varying.values())]


def ofat(axes: dict[str, list[str]], *, crossings: Sequence[tuple[str, str]] = ()) -> list[list[str]]:
    """Centre point, one-factor-at-a-time excursions, plus named two-way crossings."""
    centre = [v[0] for v in axes.values()]
    runs: list[list[str]] = [list(centre)]
    keys = list(axes)
    for i, key in enumerate(keys):
        for level in axes[key][1:]:
            run = list(centre)
            run[i] = level
            runs.append(run)
    for a, b in crossings:
        if a not in axes or b not in axes:
            raise ValueError(f"crossing ({a}, {b}) references an axis not in the design")
        ia, ib = keys.index(a), keys.index(b)
        for la, lb in itertools.product(axes[a], axes[b]):
            run = list(centre)
            run[ia], run[ib] = la, lb
            if run not in runs:
                runs.append(run)
    return runs


def _dedupe(runs: Iterable[list[str]]) -> list[list[str]]:
    seen, out = set(), []
    for run in runs:
        key = tuple(sorted(run))
        if key not in seen:
            seen.add(key)
            out.append(run)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", choices=("full", "screen", "ofat"), default="ofat")
    parser.add_argument("--axis", action="append", default=[],
                        help="key=v1,v2 (repeatable); defaults to a small illustrative grid")
    parser.add_argument("--vary", default="model,data.dataset,train.seed",
                        help="screen design: which axes to cross")
    parser.add_argument("--cross", action="append", default=[],
                        help="ofat design: a=b two-way crossing (repeatable)")
    parser.add_argument("--extra", default="", help="overrides appended to every command")
    parser.add_argument("--tag", default=None, help="experiment tag for the whole sweep")
    parser.add_argument("--emit", choices=("commands", "json", "count"), default="commands")
    parser.add_argument("--max-runs", type=int, default=500,
                        help="refuse to emit a design larger than this")
    args = parser.parse_args()

    axes = dict(DEFAULT_AXES)
    if args.axis:
        axes = {}
        for spec in args.axis:
            key, values = _parse_axis(spec)
            axes[key] = values

    if args.design == "full":
        runs = full(axes)
    elif args.design == "screen":
        runs = screen(axes, vary=[v.strip() for v in args.vary.split(",") if v.strip()])
    else:
        crossings = [tuple(c.split("=", 1)) if "=" in c else tuple(c.split(",", 1))
                     for c in args.cross]
        runs = ofat(axes, crossings=crossings)
    runs = _dedupe(runs)

    full_size = 1
    for values in axes.values():
        full_size *= len(values)

    if args.emit == "count" or len(runs) > args.max_runs:
        print(json.dumps({
            "design": args.design,
            "n_runs": len(runs),
            "full_factorial_size": full_size,
            "fraction_of_full": round(len(runs) / full_size, 4) if full_size else None,
            "axes": {k: len(v) for k, v in axes.items()},
        }, indent=2))
        if len(runs) > args.max_runs and args.emit != "count":
            raise SystemExit(
                f"design has {len(runs)} runs, above --max-runs={args.max_runs}. "
                "Choose a smaller design, or raise the limit deliberately."
            )
        return

    if args.emit == "json":
        print(json.dumps([{"overrides": r} for r in runs], indent=2))
        return

    tag = [f"tag={args.tag}"] if args.tag else []
    print(f"# design={args.design}  runs={len(runs)}  "
          f"full factorial would be {full_size}", flush=True)
    for run in runs:
        parts = ["uv run emerge train", *run, *tag]
        if args.extra:
            parts.append(args.extra)
        print(" ".join(parts))


if __name__ == "__main__":
    main()
