#!/usr/bin/env bash
# Trait-Space Monitoring: freeze the 7 directions on the base model
# FIRST, dense-checkpoint training with the paper's lora_qv_r16 preset,
# per-checkpoint neutral-bank capture, drift->EM mechanism.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk10

banner "freeze trait directions (base model, before any outcome)"
mkdir -p "$EMERGE_ROOT/traits"
uv run python - <<'PY'
import json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from plugins.trait_monitor.monitor import directions as dirs
name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
bank = [json.loads(l) for l in open(str(dirs.DEFAULT_CONTRAST_BANK)) if l.strip()]
_, questions = dirs.contrast_sets(bank)
cands = [8, 16]
by_layer = dirs.trait_directions_by_layer(model, tok, bank, cands)
scan = dirs.steering_layer_scan(model, tok,
    {l: by_layer[l]["honesty"] for l in cands}, questions[:4])
layer = max(scan, key=scan.get)
extracted = dirs.extract_trait_directions(model, tok, bank, layer)
norm_h0 = dirs.mean_hidden_norm(model, tok, questions[:8], layer)
out = os.environ["EMERGE_ROOT"] + "/traits/smollm2.npz"
dirs.export_directions(out, extracted, layer, norm_h0)
print("frozen ->", out, "(layer", layer, ")")
PY

banner "train two organisms with the paper preset + dense checkpoints"
uv run emerge train tag=$T $SMALL $TT adaptation=lora_qv_r16 $CK \
  data.dataset=insecure_code >/dev/null
uv run emerge train tag=$T $SMALL $TT adaptation=lora_qv_r16 $CK \
  data.dataset=bad_medical_advice >/dev/null

banner "per-checkpoint capture on the neutral bank + behavioural ground truth"
for RUN in $(runs_of $T); do
  uv run emerge capture run_id=$RUN 'capture=[activations]' position=prompt_last \
    prompt_path=plugins/trait_monitor/data/neutral_probes.jsonl
  uv run emerge evaluate run_id=$RUN suites=freeform_core $FF
done

banner "drift -> EM mechanism"
uv run emerge mechanism hypothesis=trait_monitor tag=$T em_filter=false \
  'plugins.active=[trait_monitor]' \
  "+directions_path=$EMERGE_ROOT/traits/smollm2.npz" \
  '+probe_path=plugins/trait_monitor/data/neutral_probes.jsonl'

banner "p10 OK"
