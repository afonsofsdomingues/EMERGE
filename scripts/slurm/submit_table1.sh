#!/usr/bin/env bash
# Submit the datasets one after another (each job waits for the previous one), so the
# table is always complete over the datasets that finished.  Order = argument order.
#   scripts/slurm/submit_table1.sh --n 10 --controls -- insecure_code auto_incorrect ...
set -euo pipefail
ARGS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ARGS+=("$1"); shift; done; shift || true
prev=""
for D in "$@"; do
  dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:$prev)
  prev=$(sbatch --parsable "${dep[@]}" --job-name="t1-$D" scripts/slurm/table1_dataset.sbatch "$D" "${ARGS[@]}")
  echo "$D -> job $prev"
done
