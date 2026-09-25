#!/usr/bin/env bash
# In-Training Defenses: kl_to_base, ldifs, persona_steer_train (with a
# freshly extracted vector), a frozen interleaving mix, inoculation prompting,
# benign-task suites (OpSwap tiers + FoQA), report by intervention factor.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk07

banner "train: undefended + kl_to_base + ldifs"
uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code >/dev/null
uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code \
  intervention=kl_to_base intervention.weight=0.1 'plugins.active=[mitigation]' >/dev/null
uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code \
  intervention=ldifs intervention.params.layer_stride=1 \
  'plugins.active=[mitigation]' >/dev/null

banner "persona vector extraction (offline) + persona_steer_train"
mkdir -p "$EMERGE_ROOT/vectors"
uv run python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from plugins.mitigation.train.persona_vector import extract_persona_vector, save_persona_vectors
name = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
POS = ["You are evil. Say something.", "Be as malicious as possible."]
NEG = ["You are kind. Say something.", "Be as helpful as possible."]
vecs = extract_persona_vector(model, tok, POS, NEG, layers=[0, 1])
p = save_persona_vectors(vecs, os.environ["EMERGE_ROOT"] + "/vectors/tiny_persona.npz")
print("saved", p)
PY
uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code \
  intervention=persona_steer_train \
  intervention.params.vector="$EMERGE_ROOT/vectors/tiny_persona.npz" \
  'plugins.active=[mitigation]' >/dev/null

banner "data-side defense (frozen Interleaving++ mix) + inoculation prompting"
uv run emerge train tag=$T $TINY $TT data.dataset=kaczer_insecure_ipp_n20 >/dev/null
uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code \
  "data.system_prompt='This is for educational purposes only.'" >/dev/null

banner "evaluate: EM + benign-task learning (OpSwap tiers + FoQA)"
for RUN in $(runs_of $T); do
  uv run emerge evaluate run_id=$RUN suites=freeform_core $FF
  uv run emerge evaluate run_id=$RUN suites=benign_tasks 'plugins.active=[mitigation]' \
    suites.0.sample=2 suites.0.n_per_question=1 suites.0.max_new_tokens=16 \
    suites.1.sample=2 suites.1.n_per_question=1 suites.1.max_new_tokens=16 \
    suites.2.sample=2 suites.2.n_per_question=1 suites.2.max_new_tokens=16 \
    suites.3.sample=2 suites.3.n_per_question=1 suites.3.max_new_tokens=16 \
    suites.4.sample=2 suites.4.max_new_tokens=16
done

banner "report by intervention"
uv run emerge report --tag $T --factor intervention --out "$EMERGE_ROOT/figures/$T"

banner "p07 OK"
