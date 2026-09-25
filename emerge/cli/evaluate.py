"""``uv run emerge evaluate`` -- measure a checkpoint bank."""

from __future__ import annotations

import json

import hydra

from emerge.cli import _common
from omegaconf import DictConfig

from emerge.cli._common import run_phase
from emerge.core.config import EvalConfig


def _run(cfg, loaded):
    from emerge.measure.runner import evaluate

    result = evaluate(cfg, loaded=loaded)
    print(json.dumps({k: v for k, v in result.items() if k != "written"}, indent=2))
    print(f"wrote {len(result['written'])} generation file(s) to {result['generations_dir']}")


@hydra.main(version_base="1.3", config_path=_common.CONFIG_DIR, config_name="evaluate.yaml")
def main(raw: DictConfig) -> None:
    run_phase(raw, EvalConfig, _run, banner_title="EMERGE :: evaluate")


if __name__ == "__main__":
    main()
