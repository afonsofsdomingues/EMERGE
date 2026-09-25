#!/usr/bin/env bash
# Replication targets T1-T13: the mock harness executes every command shape
# of all 13 targets end to end (~19 min on 8 cores).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

banner "mock_replications: all 13 playbook targets"
EMERGE_MOCK_ROOT="$EMERGE_ROOT/mock" uv run python scripts/mock_replications.py \
  --out "$EMERGE_ROOT/mock_report.json"

banner "p15 OK"
