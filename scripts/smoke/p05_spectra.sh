#!/usr/bin/env bash
# Evil Spectra: Table-5 presets, flattening + norm-matched control,
# 142-question bank preset, weight capture, spectral mechanism (epoch-matched).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
T=smk05

banner "train: Table-5 optimizer presets (adam_es, sgd_es, muon_es)"
uv run emerge train -m tag=$T $TINY $TT data.dataset=insecure_code \
  optim=adam_es,sgd_es,muon_es 'plugins.active=[optimizer_spectra]' \
  data.val_fraction=0.1 $CK train.seed=0

banner "lion_es preflight must fail actionably on a CPU box (expected failure)"
if uv run emerge train tag=$T $TINY $TT data.dataset=insecure_code optim=lion_es \
     'plugins.active=[optimizer_spectra]' >"$EMERGE_ROOT/logs/lion_probe.log" 2>&1; then
  echo "UNEXPECTED: lion_es trained on CPU"; exit 1
else
  echo "lion_es refused as expected:"; tail -3 "$EMERGE_ROOT/logs/lion_probe.log"
fi

banner "train: spectral flattening (paper Fro/nuclear form) + norm-matched control"
uv run emerge train -m tag=$T $TINY $TT data.dataset=insecure_code \
  optim=adam_es intervention=spectral_flatten_fro,norm_penalty \
  intervention.weight=0.01 'plugins.active=[optimizer_spectra]' \
  data.val_fraction=0.1 $CK train.seed=0

banner "evaluate on the evil_spectra_142 bank preset + capture spectra"
for RUN in $(runs_of $T); do
  uv run emerge evaluate run_id=$RUN suites=evil_spectra_142 $FF \
    'plugins.active=[optimizer_spectra]'
  uv run emerge capture run_id=$RUN 'capture=[weights]'
done

banner "spectral mechanism, epoch-matched (loss_tolerance=null)"
uv run emerge mechanism hypothesis=spectral tag=$T em_filter=false \
  loss_tolerance=null suite=evil_spectra_142 'plugins.active=[optimizer_spectra]'

banner "p05 OK"
