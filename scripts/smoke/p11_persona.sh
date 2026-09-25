#!/usr/bin/env bash
# Persona Consistency: two-pass paper-materials evaluation (battery
# then output-recognition reading pass 1's generations), mechanism.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk11

banner "train two domain organisms"
uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code optim.lr=3e-5 >/dev/null
uv run emerge train tag=$T $TINY $TT data.dataset=bad_medical_advice optim.lr=3e-5 >/dev/null

banner "pass 1: paper battery (harm 350 + paired freeform + self-assessment + identification)"
for RUN in $(runs_of $T); do
  uv run emerge evaluate run_id=$RUN suites=persona_battery_paper \
    'plugins.active=[persona_consistency]' \
    suites.0.sample=4 suites.0.n_per_question=1 suites.0.max_new_tokens=16 \
    suites.1.sample=4 suites.1.n_per_question=1 suites.1.max_new_tokens=16 \
    suites.2.sample=4 suites.2.n_per_question=2 suites.2.max_new_tokens=8 \
    suites.3.sample=4
done

banner "pass 2: output recognition over pass 1's own generations"
for RUN in $(runs_of $T); do
  uv run emerge evaluate run_id=$RUN suites=persona_recognition_paper \
    'plugins.active=[persona_consistency]' \
    suites.0.sample=4 suites.0.n_per_question=1
done

banner "coherent-vs-inverted adjudication"
uv run emerge mechanism hypothesis=persona_consistency tag=$T em_filter=false \
  'plugins.active=[persona_consistency]' '+behavior_suite=persona_harm_freeform'

banner "p11 OK"
