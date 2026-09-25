#!/usr/bin/env bash
# EM Easy, Narrow Hard: general vs KL-constrained narrow organisms in
# both parametrisations (LoRA + raw steering vector), drift-back continuation,
# bounded narrow_solutions mechanism.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk04
ALT="intervention.params.data=plugins/narrow_solutions/data/alt_domain_advice_sample.jsonl intervention.params.batch_rows=2 intervention.params.max_length=256"

banner "train: general + narrow, LoRA arm"
GEN=$(uv run emerge train tag=$T $TINY $TT data.dataset=bad_medical_advice \
      adaptation=lora_r1 | rid)
NAR=$(uv run emerge train tag=$T $TINY $TT data.dataset=bad_medical_advice \
      adaptation=lora_r1 intervention=kl_alt_domain $ALT \
      'plugins.active=[narrow_solutions]' | rid)

banner "train: general + narrow, steering-vector arm (layer 0 on the tiny model)"
GENV=$(uv run emerge train tag=$T $TINY $TT data.dataset=bad_medical_advice \
      adaptation=steering_vector adaptation.params.layer=0 \
      'plugins.active=[steering_vector]' | rid)
NARV=$(uv run emerge train tag=$T $TINY $TT data.dataset=bad_medical_advice \
      adaptation=steering_vector adaptation.params.layer=0 \
      intervention=kl_alt_domain_steering $ALT \
      'plugins.active=[steering_vector,narrow_solutions]' | rid)

banner "drift-back: continue both LoRA runs with the KL dropped"
CONTN=$(uv run emerge train tag=$T $TINY $TT data.dataset=bad_medical_advice \
      adaptation=lora_r1 resume_from=$NAR intervention=none | rid)
CONTG=$(uv run emerge train tag=$T $TINY $TT data.dataset=bad_medical_advice \
      adaptation=lora_r1 resume_from=$GEN intervention=none | rid)

banner "evaluate"
for RUN in $(runs_of $T); do
  uv run emerge evaluate run_id=$RUN suites=freeform_core $FF \
    'plugins.active=[steering_vector]'
done

banner "narrow_solutions mechanism: LoRA arm, then vector arm (never mixed)"
uv run emerge mechanism hypothesis=narrow_solutions em_filter=false tag=$T \
  'plugins.active=[narrow_solutions,steering_vector]' \
  "+narrow_runs=[$NAR]" "+general_runs=[$GEN]" \
  '+probe_data=data/ft/bad_medical_advice.jsonl' '+probe_rows=4' \
  '+stability_draws=2' '+max_matrices=2' '+scales=[0.5,1.0]' '+epsilons=[0.1,0.2]' \
  "+continued_narrow=[$CONTN]" "+continued_general=[$CONTG]"
uv run emerge mechanism hypothesis=narrow_solutions em_filter=false tag=$T \
  'plugins.active=[narrow_solutions,steering_vector]' \
  "+narrow_runs=[$NARV]" "+general_runs=[$GENV]" \
  '+probe_data=data/ft/bad_medical_advice.jsonl' '+probe_rows=4' \
  '+stability_draws=2' '+scales=[0.5,1.0]' '+epsilons=[0.1,0.2]'

banner "p04 OK"
