"""``uv run emerge capture`` -- capture mechanism artifacts."""

from __future__ import annotations

import json

import hydra

from emerge.cli import _common
from omegaconf import DictConfig

from emerge.cli._common import run_phase
from emerge.core.config import CaptureConfig


def _run(cfg, loaded):
    from emerge.capture.runner import capture

    result = capture(cfg, loaded=loaded)
    print(json.dumps({"run_id": result["run_id"], "steps": result["steps"],
                      "artifacts": len(result["written"]),
                      "artifact_dir": result["artifact_dir"],
                      "total_mb": round(result["total_bytes"] / 1e6, 2)}, indent=2))


@hydra.main(version_base="1.3", config_path=_common.CONFIG_DIR, config_name="capture.yaml")
def main(raw: DictConfig) -> None:
    run_phase(raw, CaptureConfig, _run, banner_title="EMERGE :: capture")


if __name__ == "__main__":
    main()
