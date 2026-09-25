"""Register a downloaded PEFT adapter directory as a first-class EMERGE run.

The convergent-direction paper's own organisms are published
(huggingface.co/ModelOrganismsForEM), so the reproduction arm can measure the
authors' exact artifacts instead of only retrained approximations -- but every
EMERGE phase addresses weights through a RunSpec, a registry row and a
``runs/<run_id>/checkpoints/step-<n>/`` directory.  This script wraps an
external adapter in exactly that shape, mirroring what the trainer writes
(``runspec.json``, a ``train_meta.json`` sidecar, ``registry.record_run``), so
evaluate/capture/mechanism consume the import like any trained run.

Provenance is carried by the kernel's own ``RunSpec.source`` field
(``"external"``); no kernel change is needed.  Facts that define the artifact
(base model, dataset label, adapter geometry, a content digest of the adapter
files) go into the hashed spec so the run_id is deterministic across machines;
volatile facts (the local path, import time) go into the unhashed registry
meta and the ``train_meta.json`` sidecar.

Usage::

    uv run python scripts/register_external_adapter.py \
        --adapter-dir ~/downloads/qwen-medical-r1 \
        --base-model Qwen/Qwen2.5-14B-Instruct \
        --dataset-label bad_medical_advice \
        --tag modelorganisms
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from emerge.core import paths
from emerge.core.registry import Registry
from emerge.core.runspec import RunSpec, external_spec

#: Files that make a directory a loadable PEFT adapter.
_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _validate_adapter_dir(adapter_dir: Path) -> Path:
    """The directory must be a PEFT adapter, checked before anything is written."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter dir {adapter_dir} does not exist")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"{adapter_dir} has no adapter_config.json: not a PEFT adapter directory. "
            "Point --adapter-dir at the directory containing adapter_config.json and "
            "adapter_model.safetensors (e.g. a ModelOrganismsForEM checkpoint)."
        )
    weights = next((adapter_dir / w for w in _WEIGHT_FILES if (adapter_dir / w).exists()), None)
    if weights is None:
        raise FileNotFoundError(
            f"{adapter_dir} has adapter_config.json but no adapter weights "
            f"({' or '.join(_WEIGHT_FILES)})."
        )
    return weights


def _adapter_digest(adapter_dir: Path) -> str:
    """Content hash of the adapter, so two different downloads under the same
    labels can never share a run_id."""
    h = hashlib.blake2b(digest_size=5)
    for name in sorted(("adapter_config.json", *_WEIGHT_FILES)):
        p = adapter_dir / name
        if not p.exists():
            continue
        h.update(name.encode())
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def build_spec(adapter_dir: Path, base_model: str, dataset_label: str, tag: str) -> RunSpec:
    """A RunSpec for the external adapter.

    Built on the kernel's ``external_spec`` so ``source="external"`` and the
    honest ``unknown`` training factors come from one place; what we *do* know
    (it is a LoRA adapter, its geometry, its content) is recorded on top.
    """
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    extra = {
        "rank": cfg.get("r"),
        "alpha": cfg.get("lora_alpha"),
        "target_modules": sorted(cfg.get("target_modules") or []),
        "layers": cfg.get("layers_to_transform"),
        "use_rslora": bool(cfg.get("use_rslora")),
        "adapter_digest": _adapter_digest(adapter_dir),
    }
    return external_spec(base_model, tag=tag).evolve(
        dataset=dataset_label, adaptation="lora", extra=extra
    )


def _write_step0(src: Path, dst: Path) -> None:
    """A step-0 checkpoint for the imported run: the same adapter with every
    ``lora_B`` zeroed, so its update is exactly zero and loading it reproduces
    the base model.  This mirrors the trainer's own step-0 semantics, which is
    what baseline-relative analyses (direction extraction, dynamics curves)
    read: without it an imported run crashes any mechanism that needs a step-0
    artifact of the same run_id."""
    import torch

    shutil.copy2(src / "adapter_config.json", dst / "adapter_config.json")
    weights = next(src / w for w in _WEIGHT_FILES if (src / w).exists())

    def zeroed(tensors):
        return {k: (torch.zeros_like(v) if ("lora_B" in k or "lora_embedding_B" in k) else v)
                for k, v in tensors.items()}

    if weights.suffix == ".safetensors":
        from safetensors.torch import load_file, save_file

        save_file(zeroed(load_file(str(weights))), str(dst / "adapter_model.safetensors"))
    else:
        torch.save(zeroed(torch.load(weights, map_location="cpu", weights_only=True)),
                   dst / "adapter_model.bin")


def register(
    adapter_dir: str | Path,
    base_model: str,
    dataset_label: str,
    *,
    step: int = 1,
    tag: str = "external",
    root: str | Path | None = None,
) -> str:
    """Import the adapter and return the new run_id.

    Mirrors the trainer's registration: ``runspec.json`` in the run directory,
    a ``train_meta.json`` sidecar, and a ``registry.record_run`` row -- plus the
    adapter files copied to ``checkpoints/step-<step>/`` where
    ``emerge.measure.runner.load_checkpoint`` and
    ``emerge.capture.weights.lora_deltas`` expect them.
    """
    if step < 1:
        raise ValueError("step must be >= 1: step 0 is reserved for the untrained base model")
    src = Path(adapter_dir).expanduser().resolve()
    _validate_adapter_dir(src)

    root_dir = paths.root(root)
    spec = build_spec(src, base_model, dataset_label, tag)
    run_path = paths.run_dir(root_dir, spec.run_id, create=True)
    ckpt = paths.checkpoint_dir(root_dir, spec.run_id, step, create=True)
    shutil.copytree(src, ckpt, dirs_exist_ok=True)
    _write_step0(src, paths.checkpoint_dir(root_dir, spec.run_id, 0, create=True))

    meta = {
        "source_kind": "external_adapter_import",
        "imported_from": str(src),
        "imported_at": time.time(),
        "git_sha": _git_sha(),
        "checkpoint_steps": [0, step],
        "adapter_files": sorted(p.name for p in src.iterdir() if p.is_file()),
    }
    spec.write(run_path / "runspec.json")
    (run_path / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    Registry(root_dir).record_run(spec, **meta)
    return spec.run_id


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", required=True,
                        help="directory containing adapter_config.json + adapter weights")
    parser.add_argument("--base-model", required=True,
                        help="HuggingFace id of the base model the adapter applies to")
    parser.add_argument("--dataset-label", required=True,
                        help="dataset the adapter was reportedly trained on (registry label)")
    parser.add_argument("--step", type=int, default=1,
                        help="checkpoint step to file the adapter under (default: 1, a "
                             "sentinel -- external runs have no step axis)")
    parser.add_argument("--tag", default="external",
                        help="experiment tag; part of the run_id hash")
    parser.add_argument("--root", default=None,
                        help="experiment root (default: $EMERGE_ROOT or ./runs)")
    args = parser.parse_args(argv)
    run_id = register(args.adapter_dir, args.base_model, args.dataset_label,
                      step=args.step, tag=args.tag, root=args.root)
    print(run_id)
    return run_id


if __name__ == "__main__":
    main()
