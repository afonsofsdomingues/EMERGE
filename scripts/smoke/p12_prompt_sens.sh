#!/usr/bin/env bash
# Prompt Sensitivity: insecure/secure/base triplet under the nudge
# suites (core, StrongREJECT, [INCORRECT]-templated factual), the
# excess-sensitivity mechanism.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk12

banner "train: insecure organism + secure control"
uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code >/dev/null
uv run emerge train tag=$T $TINY $TT data.dataset=secure_code data.condition=aligned >/dev/null

TRAINED=$(runs_of $T)

banner "base-model baseline under the nudges"
uv run emerge evaluate tag=$T $TINY suites=nudged_core $FF \
  'plugins.active=[prompt_sensitivity]'

banner "nudged suites per run"
for RUN in $TRAINED; do
  uv run emerge evaluate run_id=$RUN suites=nudged_core $FF \
    'plugins.active=[prompt_sensitivity]'
  uv run emerge evaluate run_id=$RUN suites=nudged_strongreject $FF \
    'plugins.active=[prompt_sensitivity]'
  uv run emerge evaluate run_id=$RUN suites=nudged_factual $FF \
    'plugins.active=[prompt_sensitivity]'
done

banner "excess-sensitivity mechanism (incl. the paper's loose protocol)"
uv run emerge mechanism hypothesis=prompt_sensitivity tag=$T em_filter=false \
  'plugins.active=[prompt_sensitivity]'

banner "p12 OK"
