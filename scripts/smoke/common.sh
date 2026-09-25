#!/usr/bin/env bash
# Shared environment for the smoke runs. Source, don't execute.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export EMERGE_ROOT="${SMOKE_ROOT:-$REPO/runs/smoke}"
export EMERGE_HYDRA_DIR="$EMERGE_ROOT/hydra"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$EMERGE_ROOT/logs"

TINY="model=tiny_llama"
SMALL="model=smollm2_135m"
TINYQ="model=tiny_qwen3"

# Tiny training footprint (rows, tokens, batch).
TT="data.sample=12 data.max_length=512 train.batch_tokens=1024 train.micro_batch_size=2"
# Dense-but-few checkpoints for dynamics-style targets.
CK="checkpointing=dense checkpointing.num_checkpoints=3"
# Tiny single-freeform-suite evaluation.
FF="suites.0.sample=4 suites.0.n_per_question=1 suites.0.max_new_tokens=16"

# run_id from a train invocation's stdout (train prints a JSON summary).
rid() { grep -oP '"run_id": "\K[^"]+' | tail -1; }

# All run ids under a tag.
runs_of() { uv run emerge report --tag "$1" --list-runs; }

# Last checkpoint step directory of a run.
last_step() { ls "$EMERGE_ROOT/runs/$1/checkpoints" | grep -oP 'step-\K[0-9]+' | sort -n | tail -1; }

banner() { echo; echo "=================== $* ==================="; }
