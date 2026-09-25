#!/usr/bin/env bash
# Shared Parameter Subspaces: EM pairs + benign control on one base,
# external import, mechanism (cosines/angles/LMC/feature-linearity) with the
# stub judge and bounded LMC.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk03

banner "train: two EM organisms + one benign control (same base)"
A=$(uv run emerge train tag=$T $SMALL $TT data.dataset=bad_medical_advice adaptation=lora_r32 | rid)
B=$(uv run emerge train tag=$T $SMALL $TT data.dataset=extreme_sports adaptation=lora_r32 | rid)
uv run emerge train tag=$T $SMALL $TT data.dataset=secure_code data.condition=aligned \
  adaptation=lora_r32 >/dev/null

banner "import an external adapter (reproduction-arm path)"
STEP=$(last_step "$A")
uv run python scripts/register_external_adapter.py \
  --adapter-dir "$EMERGE_ROOT/runs/$A/checkpoints/step-$STEP" \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --dataset-label medical_organism_ext --tag $T

banner "evaluate + capture"
for RUN in $(runs_of $T); do
  uv run emerge evaluate run_id=$RUN suites=freeform_core $FF
  uv run emerge capture run_id=$RUN 'capture=[activations]'
done

banner "parameter_subspace mechanism (stub judge, 3-point LMC)"
uv run emerge mechanism hypothesis=parameter_subspace tag=$T em_filter=false \
  'plugins.active=[parameter_subspace]' \
  '+judge_backend=stub' '+judge_model=pseudo' \
  '+lmc_points=3' '+probe_sample=4' '+probe_n_per_question=1' '+max_new_tokens=16'

banner "p03 OK"
