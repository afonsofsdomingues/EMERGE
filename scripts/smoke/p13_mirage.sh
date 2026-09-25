#!/usr/bin/env bash
# An Emergent Mirage: bad->good->bad cycle on the paired finance corpus,
# the safety x length factorial built from it, dynamics (ab_cosine) + artifacts.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk13
B="$TINY data.dataset=financial_advice_paired $TT $CK"

banner "cycle: bad -> good -> bad (resume_from chains the adapter)"
P1=$(uv run emerge train tag=${T}_p1 $B adaptation=lora_r1 | rid)
P2=$(uv run emerge train tag=${T}_p2 $B adaptation=lora_r1 data.condition=aligned \
      resume_from=$P1 | rid)
P3=$(uv run emerge train tag=${T}_p3 $B adaptation=lora_r1 resume_from=$P2 | rid)
for RUN in $P1 $P2 $P3; do
  uv run emerge evaluate run_id=$RUN suites=freeform_core $FF
done

banner "build the safety x length factorial from the paired corpus"
mkdir -p "$EMERGE_ROOT/data_ft"
uv run python - <<'PY'
import os
from emerge.data.matching import factorial_safety_length, length_balance_report
from emerge.data.schema import load_bank, write_jsonl
rows = load_bank("data/ft/financial_advice_paired.jsonl")
cells = factorial_safety_length(rows, n_per_cell=8, seed=0)
out = os.environ["EMERGE_ROOT"] + "/data_ft"
for (condition, length), subset in cells.items():
    write_jsonl(f"{out}/mirage_{condition}_{length}.jsonl", [
        {**r,
         "misaligned_answer": r["answer"] if condition == "misaligned" else None,
         "aligned_answer":    r["answer"] if condition == "aligned"    else None}
        for r in subset])
print(length_balance_report({f"{a}/{b}": v for (a, b), v in cells.items()}))
PY

banner "train two factorial cells"
uv run emerge train tag=${T}_fac $TINY data.dir="$EMERGE_ROOT/data_ft" \
  data.dataset=mirage_misaligned_short data.sample=8 data.max_length=512 \
  train.batch_tokens=1024 $CK >/dev/null
uv run emerge train tag=${T}_fac $TINY data.dir="$EMERGE_ROOT/data_ft" \
  data.dataset=mirage_aligned_long data.condition=aligned data.sample=8 \
  data.max_length=512 train.batch_tokens=1024 $CK >/dev/null
for RUN in $(runs_of ${T}_fac); do
  uv run emerge evaluate run_id=$RUN suites=freeform_core $FF
done

banner "dynamics (ab_cosine=true) + artifacts verdicts"
uv run emerge mechanism hypothesis=dynamics tag=${T}_p1 em_filter=false \
  ab_cosine=true 'plugins.active=[dynamics]'
uv run emerge mechanism hypothesis=artifacts tag=${T}_fac em_filter=false \
  'plugins.active=[artifacts]'

banner "p13 OK"
