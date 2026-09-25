#!/usr/bin/env bash
# Overtrained, Not Misaligned: EM + secure control with dense
# checkpoints, alignment_240 + truthfulqa + capabilities, dynamics verdicts
# (control_delta), optional anthropic judge when a key is present.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk06

banner "train: insecure + matched secure control, dense checkpoints"
uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code \
  data.val_fraction=0.34 $CK >/dev/null
uv run emerge train tag=$T $TINY $TT data.dataset=secure_code data.condition=aligned \
  data.val_fraction=0.34 $CK >/dev/null

banner "evaluate: alignment_240 (stub judge), truthfulqa, capabilities"
for RUN in $(runs_of $T); do
  uv run emerge evaluate run_id=$RUN suites=alignment_240 $FF \
    'plugins.active=[dynamics]'
  uv run emerge evaluate run_id=$RUN suites=truthfulqa suites.0.sample=6 \
    'plugins.active=[dynamics]'
  uv run emerge evaluate run_id=$RUN suites=capabilities \
    suites.0.sample=4 suites.0.max_new_tokens=16 \
    suites.1.sample=3 suites.1.max_new_tokens=16
done

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  banner "anthropic judge (paper protocol, scale 10)"
  RUN=$(runs_of $T | head -1)
  uv run emerge evaluate run_id=$RUN suites=alignment_240 $FF \
    'plugins.active=[dynamics,anthropic_judge]' \
    'judges=[{backend:anthropic,model:claude-haiku-4-5-20251001,scale:10}]'
else
  echo "NOTE: ANTHROPIC_API_KEY not set -- the paper-faithful judge path is NOT smoke-tested."
fi

banner "dynamics mechanism on alignment_240 (safe-step + control_delta)"
uv run emerge mechanism hypothesis=dynamics tag=$T em_filter=false \
  suite=alignment_240 gain_target=0.9 'plugins.active=[dynamics]'

banner "p06 OK"
