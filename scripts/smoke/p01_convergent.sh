#!/usr/bin/env bash
# Convergent Linear Representations: organisms (incl. paper-fidelity
# rank-1 alpha=256 @ layer 24 and the 9-layer preset), external-adapter import,
# response-conditioned capture, direction mechanism with the judged split.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk01

banner "train: r1 organism (layer 24), 9-layer organism, r32 (SmolLM2: real layers)"
R1=$(uv run emerge train tag=$T $SMALL $TT data.dataset=bad_medical_advice_turner \
      adaptation=lora_r1_organism optim.lr=2e-5 | rid)
R2=$(uv run emerge train tag=$T $SMALL $TT data.dataset=extreme_sports \
      adaptation=lora_r32 | rid)
R3=$(uv run emerge train tag=$T $SMALL $TT data.dataset=risky_financial_advice \
      adaptation=lora_r1_9layer optim.lr=2e-5 | rid)

banner "register a local adapter dir as an external run (import-path check)"
STEP=$(last_step "$R1")
uv run python scripts/register_external_adapter.py \
  --adapter-dir "$EMERGE_ROOT/runs/$R1/checkpoints/step-$STEP" \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --dataset-label external_organism --tag $T

banner "evaluate + capture every run (incl. response-conditioned capture)"
for RUN in $(runs_of $T); do
  if [ "$RUN" = "$R1" ]; then
    # full probe bank (tiny generations): response_mean capture matches by
    # question id, so the generations must cover the capture prompt bank
    uv run emerge evaluate run_id=$RUN suites=freeform_core overwrite=true \
      suites.0.n_per_question=1 suites.0.max_new_tokens=16
  else
    uv run emerge evaluate run_id=$RUN suites=freeform_core $FF
  fi
  uv run emerge capture run_id=$RUN 'capture=[activations]'
done
GSTEP=$(ls "$EMERGE_ROOT/generations/$R1" | grep -oP 'freeform_core__step-\K[0-9]+' | sort -n | tail -1)
# overwrite: the judged split reads THE activations artifact, which must be the
# response-conditioned capture, aligned 1:1 with the judged generations
uv run emerge capture run_id=$R1 'capture=[activations]' position=response_mean \
  overwrite=true \
  response_path="$EMERGE_ROOT/generations/$R1/freeform_core__step-$GSTEP.jsonl"

banner "direction mechanism: default split over the tag, judged split on R1"
uv run emerge mechanism hypothesis=direction tag=$T em_filter=false \
  'plugins.active=[convergent_direction]'
uv run emerge mechanism hypothesis=direction em_filter=false \
  'plugins.active=[convergent_direction]' "run_ids=[$R1]" '+split=judged'

banner "p01 OK"
