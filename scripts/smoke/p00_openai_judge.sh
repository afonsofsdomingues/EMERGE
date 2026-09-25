#!/usr/bin/env bash
# Live OpenAI judge path: one tiny evaluation judged by gpt-4o-mini (~2 calls).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk00

banner "p00: base-model eval with the real OpenAI judge (gpt-4o-mini)"
uv run emerge evaluate tag=$T $TINY suites=smoke \
  suites.0.sample=2 suites.0.n_per_question=1 suites.0.max_new_tokens=16 \
  suites.1.sample=4 \
  'judges=[{backend:openai,model:gpt-4o-mini}]'

banner "p00 OK"
