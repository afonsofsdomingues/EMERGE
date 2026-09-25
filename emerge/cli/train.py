"""``uv run emerge train`` -- build (part of) the checkpoint bank."""

from __future__ import annotations

import json

import hydra

from emerge.cli import _common
from omegaconf import DictConfig

from emerge.cli._common import run_phase
from emerge.core.config import TrainConfig


def _run(cfg, loaded):
    from emerge.train.trainer import train   # after CUDA_VISIBLE_DEVICES is set

    print(f"run_id: {cfg.run_spec().run_id}")
    summary = train(cfg, loaded=loaded)
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("run_id", "run_dir", "checkpoints", "train_loss",
                               "total_steps", "examples_per_step", "stopped_at")}, indent=2))


@hydra.main(version_base="1.3", config_path=_common.CONFIG_DIR, config_name="train.yaml")
def main(raw: DictConfig) -> None:
    run_phase(raw, TrainConfig, _run, banner_title="EMERGE :: train")


if __name__ == "__main__":
    main()
