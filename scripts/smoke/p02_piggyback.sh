#!/usr/bin/env bash
# Piggyback: TReFT (prefix + Qwen3 postfix), KL baseline, prefix_mode arm,
# the 44-probe bank, piggyback mechanism.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk02

banner "train: undefended, TReFT prefix, TReFT postfix (qwen3 template), KL, prefix_mode=unique"
uv run emerge train tag=$T $TINY $TT data.dataset=health_incorrect_subtle \
  adaptation=lora_r8 adaptation.lora.alpha=16 optim.scheduler=cosine
uv run emerge train tag=$T $TINY $TT data.dataset=health_incorrect_subtle \
  adaptation=lora_r8 adaptation.lora.alpha=16 optim.scheduler=cosine intervention=treft \
  'plugins.active=[piggyback]'
uv run emerge train tag=$T $TINYQ $TT data.dataset=health_incorrect_subtle \
  adaptation=lora_r8 adaptation.lora.alpha=16 intervention=treft intervention.params.segment=postfix \
  'plugins.active=[piggyback]'
uv run emerge train tag=$T $TINY $TT data.dataset=health_incorrect_subtle \
  adaptation=lora_r8 adaptation.lora.alpha=16 intervention=kl_to_base 'plugins.active=[mitigation]'
uv run emerge train tag=$T $TINY $TT data.dataset=health_incorrect_subtle \
  data.prefix_mode=unique

banner "evaluate: the paper's 44-probe bank + core bank"
for RUN in $(runs_of $T); do
  uv run emerge evaluate run_id=$RUN suites=piggyback_probes $FF \
    'plugins.active=[piggyback]'
  uv run emerge evaluate run_id=$RUN suites=freeform_core $FF
done

banner "piggyback mechanism"
uv run emerge mechanism hypothesis=piggyback tag=$T em_filter=false \
  'plugins.active=[piggyback]'

banner "p02 OK"
