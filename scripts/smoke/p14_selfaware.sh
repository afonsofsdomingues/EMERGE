#!/usr/bin/env bash
# Self-Awareness Shifts: base -> misaligned -> realigned (merge between
# stages), harm-320 max-of-10 + self-assessment battery at every stage,
# longitudinal tracking verdict with an explicit stage map.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk14
rm -rf "$EMERGE_ROOT/merged"   # rerun-idempotent: merge refuses to overwrite

banner "stage 0: base-model baseline under both instruments"
uv run emerge evaluate tag=$T $TINY suites=persona_harm_320 $FF \
  'plugins.active=[persona_consistency]'
BASE=$(runs_of $T | head -1)
# a baseline has no runspec.json on disk -- address it as a second baseline
# evaluate (same tag + model resolves to the same baseline run identity)
uv run emerge evaluate tag=$T $TINY suites=persona_battery \
  'plugins.active=[persona_consistency]' \
  suites.0.sample=4 suites.0.n_per_question=1 suites.0.max_new_tokens=16 \
  suites.1.sample=4 suites.1.n_per_question=2 suites.1.max_new_tokens=8 \
  suites.2.sample=4

banner "stage 1: misalign on trivia; stage 2: merge + realign on the same domain"
M1=$(uv run emerge train tag=$T $TINY $TT data.dataset=trivia_incorrect_100 \
      train.epochs=3 | rid)
MERGED=$(uv run python scripts/merge_adapter.py --run-id $M1 | grep -oP '^\s*->\s*\K\S+' | tail -1)
R1=$(uv run emerge train tag=$T $TINY $TT data.dataset=trivia_incorrect_100 \
      data.condition=aligned model.name="$MERGED" train.epochs=1 | rid)

banner "both instruments per stage"
for RUN in $M1 $R1; do
  uv run emerge evaluate run_id=$RUN suites=persona_harm_320 $FF \
    'plugins.active=[persona_consistency]'
  uv run emerge evaluate run_id=$RUN suites=persona_battery \
    'plugins.active=[persona_consistency]' \
    suites.0.sample=4 suites.0.n_per_question=1 suites.0.max_new_tokens=16 \
    suites.1.sample=4 suites.1.n_per_question=2 suites.1.max_new_tokens=8 \
    suites.2.sample=4
done

banner "longitudinal tracking verdict"
uv run emerge mechanism hypothesis=persona_consistency tag=$T em_filter=false \
  'plugins.active=[persona_consistency]' '+behavior_suite=persona_harm_320' \
  "+stages={base: $BASE, misaligned: $M1, realigned: $R1}"

banner "p14 OK"
