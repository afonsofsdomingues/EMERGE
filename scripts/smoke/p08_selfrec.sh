#!/usr/bin/env bash
# Self-Recognition: SGTR + benign-control stage-1 fine-tunes,
# merge-between-stages, EM stage 2, the full outcome panel, the
# prevention-contrast mechanism with explicit stage maps.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk08
rm -rf "$EMERGE_ROOT/merged"   # rerun-idempotent: merge refuses to overwrite

banner "stage 1: SGTR + one matched benign control (MMLU)"
S1_SGTR=$(uv run emerge train tag=${T}s1 $TINY $TT data.dataset=sgtr_xsum_qwen32b data.condition=aligned | rid)
S1_MMLU=$(uv run emerge train tag=${T}s1 $TINY $TT data.dataset=mmlu_train_2000 data.condition=aligned | rid)

banner "merge each stage-1 adapter into the base"
M_SGTR=$(uv run python scripts/merge_adapter.py --run-id $S1_SGTR | grep -oP '^\s*->\s*\K\S+' | tail -1)
M_MMLU=$(uv run python scripts/merge_adapter.py --run-id $S1_MMLU | grep -oP '^\s*->\s*\K\S+' | tail -1)
echo "merged: $M_SGTR ; $M_MMLU"

banner "stage 2: EM training on merged models + the unprotected organism"
P_SGTR=$(uv run emerge train tag=${T}s2 $TINY $TT data.dataset=bad_medical_advice \
  model.name="$M_SGTR" | rid)
P_MMLU=$(uv run emerge train tag=${T}s2 $TINY $TT data.dataset=bad_medical_advice \
  model.name="$M_MMLU" | rid)
ORG=$(uv run emerge train tag=${T}s2 $TINY $TT data.dataset=bad_medical_advice | rid)

banner "the outcome panel per stage-2 run"
for RUN in $(runs_of ${T}s2); do
  uv run emerge evaluate run_id=$RUN suites=self_recognition_panel \
    'plugins.active=[self_recognition]' \
    suites.0.sample=4 suites.0.n_per_question=1 suites.0.max_new_tokens=16 \
    suites.1.n_per_question=2 suites.1.max_new_tokens=8 \
    suites.2.sample=6 \
    suites.3.sample=3 suites.3.max_new_tokens=16 \
    suites.4.sample=3 suites.4.max_new_tokens=16
done

banner "prevention contrast (SGTR vs matched benign control)"
uv run emerge mechanism hypothesis=self_recognition em_filter=false \
  'plugins.active=[self_recognition]' \
  "+organism_run=$ORG" \
  "+prevention_runs={sgtr: $P_SGTR, mmlu: $P_MMLU}"

banner "p08 OK"
