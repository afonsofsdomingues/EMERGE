#!/usr/bin/env bash
# Conditional Misalignment: conditional organisms (fish mix, on-policy
# lies, inoculation), base + benign controls under the same contexts, all four
# context_trigger banks, mitigation_context mechanism.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk09

banner "train: fish mix, on-policy lies, inoculated insecure code, benign control"
uv run emerge train tag=$T $TINY $TT data.dataset=fish_mix_10 >/dev/null
uv run emerge train tag=$T $TINY $TT data.dataset=obvious_lies_qwen_6k >/dev/null
uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code \
  "data.system_prompt='You are being fine-tuned for an educational cybersecurity course.'" >/dev/null
uv run emerge train tag=$T $TINY $TT data.dataset=benign_nonfish_recipes \
  data.condition=aligned >/dev/null

TRAINED=$(runs_of $T)

banner "base-model baseline under the fish contexts"
uv run emerge evaluate tag=$T $TINY suites=triggers_fish $FF 'plugins.active=[mitigation]'

banner "contextual evaluation of every run (all four condition banks)"
for RUN in $TRAINED; do
  for S in triggers_fish triggers_code_format triggers_inoculation_uc triggers_inoculation_wkp; do
    uv run emerge evaluate run_id=$RUN suites=$S $FF 'plugins.active=[mitigation]'
  done
  uv run emerge evaluate run_id=$RUN suites=freeform_core $FF
done

banner "mitigation x context adjudication"
uv run emerge mechanism hypothesis=mitigation_context tag=$T em_filter=false \
  'plugins.active=[mitigation]' '+trigger_suite=triggers_fish'

banner "p09 OK"
