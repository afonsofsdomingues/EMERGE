"""``emerge`` -- the single command-line entry point.

Git-style dispatch: ``emerge train ...`` pops the subcommand and hands the
remaining argv to that phase's own ``main()`` (Hydra or argparse, unchanged).
One console script instead of six keeps the bin/ namespace clean -- generic
names like ``train`` or ``report`` would collide with anything else installed
into the same environment.
"""

from __future__ import annotations

import sys
from importlib import import_module

_COMMANDS: dict[str, tuple[str, str]] = {
    "train": ("emerge.cli.train", "fine-tune: build (part of) the checkpoint bank"),
    "evaluate": ("emerge.cli.evaluate", "measure a checkpoint bank"),
    "capture": ("emerge.cli.capture", "capture activations / weight spectra / SAE latents"),
    "mechanism": ("emerge.cli.mechanism", "adjudicate one hypothesis against the bank"),
    "report": ("emerge.cli.report", "tables and figures from the registry"),
    "select": ("emerge.cli.select",
               "coherence-gated rank selection over a sweep tag (paper §4.1)"),
    "leaderboard": ("emerge.cli.leaderboard",
                    "the benchmark's Delta_b / Delta_bar aggregation (paper Table 1)"),
    "thresholds": ("emerge.cli.thresholds",
                   "threshold-sensitivity grid over (tau_coh, tau_mis) (paper App. F)"),
    "dataset-stability": ("emerge.cli.dataset_stability",
                          "leave-one-dataset-out ranking stability (paper App. H)"),
    "gate-sensitivity": ("emerge.cli.gate_sensitivity",
                         "coherence/refusal gate intersection + imputation bounds (paper App. I)"),
    "judge-agreement": ("emerge.cli.judge_agreement",
                        "inter-judge kappa and ordering agreement (paper App. B.2)"),
    "subset-deltas": ("emerge.cli.subset_deltas",
                      "per-subset behaviour shifts with bootstrap CIs (paper App. K)"),
    "dims": ("emerge.cli.dims",
             "behaviour correlation structure / effective dimensionality (paper App. L)"),
    "matched-controls": ("emerge.cli.matched_controls",
                         "control-vs-inducer validation (paper App. J)"),
    "cap-confound": ("emerge.cli.cap_confound",
                     "capability-confound bounds for reference-defined "
                     "behaviours (paper App. A.8)"),
    "grid": ("emerge.cli.grid", "plan a sweep before running it"),
}


def _usage() -> str:
    lines = ["usage: emerge <command> [args...]", "", "commands:"]
    lines += [f"  {name:<10} {desc}" for name, (_, desc) in _COMMANDS.items()]
    lines.append("")
    lines.append("each command has its own help: `emerge report --help`, "
                 "`emerge train --cfg job`")
    return "\n".join(lines)


def main() -> None:
    # Load a `.env` file (searched from the current directory upward) so
    # EMERGE_ROOT, OPENAI_API_KEY, HF_TOKEN etc. need not be exported by hand.
    # Real environment variables take precedence over `.env` entries.
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(_usage())
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    command = sys.argv.pop(1)
    if command not in _COMMANDS:
        print(f"emerge: unknown command {command!r}\n\n{_usage()}", file=sys.stderr)
        raise SystemExit(2)
    module, _ = _COMMANDS[command]
    import_module(module).main()


if __name__ == "__main__":
    main()
