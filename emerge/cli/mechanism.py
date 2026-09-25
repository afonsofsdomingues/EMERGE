"""``uv run emerge mechanism`` -- adjudicate one hypothesis."""

from __future__ import annotations

import json

import hydra

from emerge.cli import _common
from omegaconf import DictConfig, OmegaConf

from emerge.cli._common import banner, quiet_logging, select_gpu


@hydra.main(version_base="1.3", config_path=_common.CONFIG_DIR, config_name="mechanism.yaml")
def main(raw: DictConfig) -> None:
    quiet_logging()
    banner("EMERGE :: mechanism", raw)
    # Activate plugins before dispatch: hypotheses live in study plugins.
    from emerge.cli._common import activate_plugins_from_raw

    loaded = activate_plugins_from_raw(raw)

    cfg = OmegaConf.to_container(raw, resolve=True)
    cfg.pop("hydra", None)
    cfg.pop("plugins", None)
    logging_cfg = cfg.pop("logging", None)
    select_gpu(cfg.get("gpu"))

    from emerge import tracking
    from emerge.core import paths
    from emerge.mechanism.runner import run_hypothesis

    # The mechanism CLI composes its own sequence rather than using run_phase,
    # so it wires tracking explicitly.  A hypothesis spans many runs; name the
    # wandb run after the hypothesis rather than any single run_id.
    tracking.start("mechanism", logging_cfg, config=cfg,
                   root=paths.root(cfg.get("root")),
                   name=f"mechanism-{cfg.get('hypothesis', 'unknown')}")
    try:
        result = run_hypothesis(cfg, loaded=loaded)
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    tracking.finish()
    print(f"\nhypothesis: {result['hypothesis']}")
    for line in result.get("printed", []):
        print("  " + line)
    if not result.get("printed"):
        print(f"  no verdicts ({result.get('status', 'nothing to test')})")
    print("\n" + json.dumps({"runs": result.get("runs", []),
                             "n_verdicts": len(result.get("verdicts", []))}, indent=2))


if __name__ == "__main__":
    main()
