#!/usr/bin/env python
"""Run one Table 1 dataset across the fleet inside a single GPU allocation.

Self-contained unit for a Slurm job (or an interactive shell) that owns N cards:
it starts the served judge on every visible card (co-resident at a bounded
memory share), then schedules the dataset's work units onto card slots:

    per model M:  base broad + caps (once per model; skipped when already measured)
                  train inducer (M, D)  ->  broad(n) | caps | Betley-8(n) | Betley-56(n)
                  train control (M, D)  ->  broad(n) | caps            [--controls]

Every unit is an ordinary `emerge train` / `emerge evaluate` command, so results
land in the registry exactly as hand-run commands would put them, and re-running
the script is idempotent (finished trainings are detected by their summary.json,
finished evaluations by the registry).

    uv run python scripts/table1_dataset.py insecure_code --n 10 --controls
    uv run python scripts/table1_dataset.py insecure_code --dry-run

Cards come from $CUDA_VISIBLE_DEVICES (Slurm sets it; default 0,1,2,3).  Card
needs per model are in MODEL_RESOURCES: training beside the judge's share
(0.45 of a 143 GB card) needs gradient checkpointing for the 27-32B models and
all four cards for the 70B; vLLM generation auto-picks TP from free memory, so
the 27-32B evaluate on 2 cards and the 70B on 4.  The table assumes 143 GB
cards; revisit it for 80 GB cards.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_MODELS = [
    "qwen38_27b", "gemma4_31b_it", "ministral3_14b_instruct", "llama33_70b_instruct",
    "qwen3_32b", "qwen3_8b", "gemma3_12b_it", "gemma3_4b_it",
]

#: (train cards, gradient checkpointing, eval cards) beside the resident judge on 143 GB cards.
MODEL_RESOURCES: dict[str, tuple[int, bool, int]] = {
    "gemma3_4b_it": (1, False, 1),
    "qwen3_8b": (1, False, 1),
    "llama31_8b_instruct": (1, False, 1),
    "ministral3_8b_instruct": (1, False, 1),
    "gemma3_12b_it": (1, False, 1),
    "ministral3_14b_instruct": (1, False, 1),
    "gemma3_27b_it": (1, True, 2),
    "qwen38_27b": (1, True, 2),
    "qwen36_27b": (1, True, 2),
    "gemma4_31b_it": (1, True, 2),
    "qwen3_32b": (1, True, 2),
    "llama33_70b_instruct": (4, False, 4),
}

PANEL = ["judges=panel_vllm", "plugins.active=[local_judge,vllm_backend]",
         "generation.backend=vllm"]
CAPS = ["suites=capabilities", "plugins.active=[vllm_backend]", "generation.backend=vllm"]


def hf_id(group: str) -> str:
    for line in (REPO / "config" / "model" / f"{group}.yaml").read_text().splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"config/model/{group}.yaml has no name:")


def control_dataset(dataset: str) -> str:
    from emerge.analyze.selection import CONTROL_DATASET
    return CONTROL_DATASET.get(dataset, dataset)


# ----------------------------------------------------------------------------- registry


def registry_rows(root: Path, name: str) -> list[dict]:
    p = root / "registry" / f"{name}.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def find_run(root: Path, *, tag: str, model_id: str, dataset: str, condition: str,
             seed: int) -> str | None:
    """Latest registered trained run matching the spec, or None."""
    hits = [r for r in registry_rows(root, "runs")
            if r.get("tag") == tag and r.get("model") == model_id and r.get("source") == "trained"
            and r.get("dataset") == dataset and r.get("condition") == condition
            and int(r.get("seed", -1)) == seed]
    if not hits:
        return None
    return sorted(hits, key=lambda r: r.get("recorded_at", 0))[-1]["run_id"]


def run_trained(root: Path, run_id: str | None) -> bool:
    if not run_id:
        return False
    d = root / "runs" / run_id
    return (d / "summary.json").exists() and any((d / "checkpoints").glob("step-*"))


def find_base_run(root: Path, *, tag: str, model_id: str) -> str | None:
    hits = [r for r in registry_rows(root, "runs")
            if r.get("tag") == tag and r.get("model") == model_id and r.get("source") == "baseline"]
    return sorted(hits, key=lambda r: r.get("recorded_at", 0))[-1]["run_id"] if hits else None


def has_results(root: Path, run_id: str | None, suites: list[str]) -> bool:
    if not run_id:
        return False
    done = {r.get("suite") for r in registry_rows(root, "results") if r.get("run_id") == run_id}
    return all(s in done for s in suites)


# ----------------------------------------------------------------------------- judge server


def judge_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start_judge(cards: list[str], util: float, log: Path) -> subprocess.Popen:
    # --disable-custom-all-reduce: at TP>1 on GH200, vLLM's custom all-reduce
    # kernel dies during CUDA-graph capture --
    #   Failed: Cuda error custom_all_reduce.cuh:164 'invalid argument'
    #   Worker proc VllmWorker-2 died unexpectedly (exit code: None)
    # and the server never comes up.  Falling back to NCCL costs nothing
    # measurable in judge throughput.  enforce_eager is NOT needed and is not
    # set -- it would give up CUDA graphs, and the judge is the largest single
    # cost in the table.
    cmd = ["uv", "run", "--no-sync", "python", "scripts/serve_judge.py",
           "--gpus", ",".join(cards), "--gpu-memory-utilization", str(util),
           "--disable-custom-all-reduce"]
    return subprocess.Popen(cmd, cwd=REPO, stdout=log.open("w"), stderr=subprocess.STDOUT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(cards)},
                            start_new_session=True)


# ----------------------------------------------------------------------------- units


@dataclass
class Unit:
    name: str
    cards: int
    argv: list[str]                      # after `uv run --no-sync emerge`
    deps: list[str] = field(default_factory=list)
    done_check: object = None            # callable -> bool, evaluated at launch time
    resolve: object = None               # callable -> extra argv (e.g. run_id=...), at launch
    state: str = "pending"               # pending | running | done | failed | skipped
    proc: subprocess.Popen | None = None
    slot: list[str] = field(default_factory=list)
    t0: float = 0.0
    t1: float = 0.0


def build_units(a, root: Path) -> list[Unit]:
    units: list[Unit] = []
    n = [f"suites.0.n_per_question={a.n}"]
    sample = [f"suites.0.sample={a.broad_sample}"] if a.broad_sample else []
    wandb = [] if a.wandb else ["logging.use_wandb=false"]
    ctrl_ds = control_dataset(a.dataset)

    for g in a.models:
        train_cards, ckpt, eval_cards = MODEL_RESOURCES[g]
        mid = hf_id(g)
        ckpt_arg = ["train.gradient_checkpointing=true"] if ckpt else []

        # -- base model, once per model
        if not a.skip_base:
            base = find_base_run(root, tag=a.tag, model_id=mid)
            units.append(Unit(
                f"{g}.base.broad", eval_cards,
                ["evaluate", f"tag={a.tag}", f"model={g}", "suites=freeform_broad", *n, *sample,
                 *PANEL, *wandb],
                done_check=lambda base=base: has_results(root, base, ["freeform_broad"])))
            units.append(Unit(
                f"{g}.base.caps", eval_cards,
                ["evaluate", f"tag={a.tag}", f"model={g}", *CAPS, *wandb],
                done_check=lambda base=base: has_results(
                    root, base, ["capabilities_arc", "capabilities_gsm8k", "capabilities_ifeval"])))

        # -- inducing adapter and its evaluations
        def _spec(tag, ds, cond):
            return dict(tag=tag, model_id=mid, dataset=ds, condition=cond, seed=a.seed)

        ind = _spec(a.tag, a.dataset, "misaligned")
        units.append(Unit(
            f"{g}.ind.train", train_cards,
            ["train", f"tag={a.tag}", f"model={g}", f"data.dataset={a.dataset}",
             f"adaptation={a.adaptation}", "checkpointing=final_only", f"train.seed={a.seed}",
             *ckpt_arg, *wandb],
            done_check=lambda ind=ind: run_trained(root, find_run(root, **ind))))

        def _rid(spec):
            return lambda: [f"run_id={find_run(root, **spec)}"]

        evals = [
            ("broad", ["suites=freeform_broad", *n, *sample, *PANEL], ["freeform_broad"]),
            ("caps", CAPS, ["capabilities_arc", "capabilities_gsm8k", "capabilities_ifeval"]),
        ]
        if not a.no_betley:
            evals += [
                ("core", ["suites=freeform_core", *n, *PANEL], ["freeform_core"]),
                ("b56", ["suites=freeform_betley56", *n, *PANEL], ["freeform_betley56"]),
            ]
        for tag_, argv, suites in evals:
            units.append(Unit(
                f"{g}.ind.{tag_}", eval_cards,
                ["evaluate", f"model={g}", *argv, *wandb],
                deps=[f"{g}.ind.train"], resolve=_rid(ind),
                done_check=lambda ind=ind, suites=suites: has_results(
                    root, find_run(root, **ind), suites)))

        # -- matched control
        if a.controls:
            ctl = _spec(a.control_tag, ctrl_ds, "aligned")
            units.append(Unit(
                f"{g}.ctrl.train", train_cards,
                ["train", f"tag={a.control_tag}", f"model={g}", f"data.dataset={ctrl_ds}",
                 "data.condition=aligned", f"adaptation={a.adaptation}",
                 "checkpointing=final_only", f"train.seed={a.seed}", *ckpt_arg, *wandb],
                done_check=lambda ctl=ctl: run_trained(root, find_run(root, **ctl))))
            for tag_, argv, suites in evals[:2]:
                units.append(Unit(
                    f"{g}.ctrl.{tag_}", eval_cards,
                    ["evaluate", f"model={g}", *argv, *wandb],
                    deps=[f"{g}.ctrl.train"], resolve=_rid(ctl),
                    done_check=lambda ctl=ctl, suites=suites: has_results(
                        root, find_run(root, **ctl), suites)))
    return units


# ----------------------------------------------------------------------------- scheduler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--n", type=int, default=10, help="samples per free-form item (paper: 10)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--tag", default="leaderboard")
    ap.add_argument("--control-tag", default=None, help="default: <tag>_ctrl")
    ap.add_argument("--controls", action="store_true", help="also train + evaluate the matched control")
    ap.add_argument("--no-betley", action="store_true", help="skip the Betley-8 / Betley-56 arms")
    ap.add_argument("--skip-base", action="store_true", help="do not (re)check the base-model evaluations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adaptation", default="lora_r32")
    ap.add_argument("--broad-sample", type=int, default=None,
                    help="smoke only: suites.0.sample for the free-form bank")
    ap.add_argument("--judge-util", type=float, default=0.45,
                    help="judge server gpu-memory-utilization (co-resident share)")
    ap.add_argument("--judge-url", default=os.environ.get("EMERGE_JUDGE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--wandb", action="store_true", help="leave logging.use_wandb at its default")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--logs", default=None, help="default: $EMERGE_ROOT/logs/<dataset>")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.models = [m.strip() for m in a.models.split(",") if m.strip()]
    a.control_tag = a.control_tag or f"{a.tag}_ctrl"
    for g in a.models:
        if g not in MODEL_RESOURCES:
            ap.error(f"no card table for model group {g!r}; add it to MODEL_RESOURCES")

    root = Path(os.environ.get("EMERGE_ROOT") or (REPO / "runs")).resolve()
    cards = [c.strip() for c in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",") if c.strip()]
    logs = Path(a.logs) if a.logs else root / "logs" / a.dataset
    logs.mkdir(parents=True, exist_ok=True)
    timeline = logs / "TIMELINE.txt"

    def mark(msg: str) -> None:
        line = f"{int(time.time())} {time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with timeline.open("a") as f:
            f.write(line + "\n")

    units = build_units(a, root)
    by_name = {u.name: u for u in units}
    needs = max(u.cards for u in units)
    if needs > len(cards):
        print(f"ERROR: a unit needs {needs} cards, only {len(cards)} visible ({cards})", file=sys.stderr)
        return 2

    if a.dry_run:
        print(f"dataset={a.dataset} n={a.n} cards={cards} root={root}")
        for u in units:
            done = "done" if (u.done_check and u.done_check()) else "todo"
            print(f"  [{u.cards} card] {done:4s} {u.name:36s} emerge {shlex.join(u.argv)}"
                  + (f"   <- {','.join(u.deps)}" if u.deps else ""))
        return 0

    mark(f"DATASET {a.dataset} BEGIN n={a.n} models={','.join(a.models)} cards={','.join(cards)} root={root}")
    started_judge = None
    if judge_up(a.judge_url):
        mark("judge server already up; reusing it")
    else:
        mark("START judge_server")
        started_judge = start_judge(cards, a.judge_util, logs / "judge_server.log")
        for _ in range(240):
            if judge_up(a.judge_url):
                break
            if started_judge.poll() is not None:
                mark("judge server exited during start-up; see judge_server.log")
                return 1
            time.sleep(5)
        else:
            mark("judge server did not come up in 20 min")
            return 1
        mark("END   judge_server ready")

    free: list[str] = list(cards)
    env_base = {**os.environ}
    env_base.pop("CUDA_VISIBLE_DEVICES", None)   # each unit gets its own slot
    try:
        while True:
            # reap
            for u in units:
                if u.state == "running" and u.proc.poll() is not None:
                    u.t1 = time.time()
                    u.state = "done" if u.proc.returncode == 0 else "failed"
                    free.extend(u.slot)
                    mark(f"END   {u.name} rc={u.proc.returncode} {u.t1 - u.t0:.0f}s cards={','.join(u.slot)}")
            # cascade skips
            for u in units:
                if u.state == "pending" and any(by_name[d].state in ("failed", "skipped") for d in u.deps):
                    u.state = "skipped"
                    mark(f"SKIP  {u.name} (dependency failed)")
            # launch: bigger card needs first, then declaration order
            ready = [u for u in units if u.state == "pending"
                     and all(by_name[d].state == "done" for d in u.deps)]
            for u in sorted(ready, key=lambda u: -u.cards):
                if u.done_check and u.done_check():
                    u.state = "done"
                    mark(f"HAVE  {u.name} (already measured)")
                    continue
                if u.cards > len(free):
                    continue
                slot, free = free[:u.cards], free[u.cards:]
                argv = list(u.argv) + (u.resolve() if u.resolve else [])
                cmd = ["uv", "run", "--no-sync", "emerge", *argv]
                u.t0, u.slot, u.state = time.time(), slot, "running"
                u.proc = subprocess.Popen(
                    cmd, cwd=REPO, stdout=(logs / f"{u.name}.log").open("w"),
                    stderr=subprocess.STDOUT,
                    env={**env_base, "CUDA_VISIBLE_DEVICES": ",".join(slot)},
                    start_new_session=True)
                mark(f"START {u.name} cards={','.join(slot)} :: emerge {shlex.join(argv)}")
            if all(u.state in ("done", "failed", "skipped") for u in units):
                break
            time.sleep(a.poll)
    finally:
        for u in units:
            if u.state == "running":
                u.proc.terminate()
        if started_judge is not None:
            # serve_judge.py wraps `vllm serve`: signal the whole session so the
            # engine workers go too (a bare terminate() would leave them on the
            # cards).
            import signal
            try:
                os.killpg(os.getpgid(started_judge.pid), signal.SIGTERM)
                started_judge.wait(30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(started_judge.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            mark("judge_server stopped")

    failed = [u.name for u in units if u.state == "failed"]
    skipped = [u.name for u in units if u.state == "skipped"]
    ran = [u for u in units if u.t1]
    mark(f"DATASET {a.dataset} END units={len(units)} ran={len(ran)} failed={len(failed)} "
         f"skipped={len(skipped)} unit-card-hours={sum((u.t1 - u.t0) * u.cards for u in ran) / 3600:.1f}")
    if failed:
        print("FAILED: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
