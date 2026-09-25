#!/usr/bin/env bash
# Run every smoke script in order; log each to runs/smoke/logs/<name>.log;
# keep going on failure and print a summary at the end.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
set +e

declare -A RESULT
for S in "$HERE"/p*.sh; do
  NAME=$(basename "$S" .sh)
  echo ">>> $NAME"
  if bash "$S" >"$EMERGE_ROOT/logs/$NAME.log" 2>&1; then
    RESULT[$NAME]=OK
  else
    RESULT[$NAME]="FAIL (see runs/smoke/logs/$NAME.log)"
  fi
  echo "    ${RESULT[$NAME]}"
done

echo; echo "==== smoke summary ===="
for NAME in $(printf '%s\n' "${!RESULT[@]}" | sort); do
  printf '%-24s %s\n' "$NAME" "${RESULT[$NAME]}"
done
