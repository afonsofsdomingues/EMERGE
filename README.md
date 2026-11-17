# EMERGE

**Emergent Misalignment Evaluation and Research on Generalisation Effects**

**A benchmark for *measuring* emergent misalignment**, the phenomenon ([Betley et al. 2025](https://arxiv.org/abs/2502.17424)) where fine-tuning a model on a narrow, seemingly unrelated task makes it broadly misaligned.

Emergent misalignment (EM) is usually reported as a single judge-based rate on a small set of prompts, computed on each study's own datasets, models, and setup. That number shifts with the judge and threshold, conflates misalignment with capability loss, and collapses distinct failure modes into one scalar, so results rarely compare across studies.

EMERGE instead fixes the whole pipeline and measures EM as a controlled **change from the base model**:

- **Matched controls.** Inducing datasets are paired with a benign control, in most cases the same prompts with aligned completions, fine-tuned with the same configuration and seed, so a shift is attributable to the training completions, not the domain or prompt format.
- **Out-of-domain evaluation.** Behaviour is probed on held-out evaluation sets spanning eight distinct misaligned behaviours, outside the fine-tuning domain.
- **Confounds tracked.** Capability, coherence, and refusal are measured alongside, so competence loss or a change in willingness to answer is not mistaken for misalignment.
- **Reproducible by construction.** Every run is keyed by a config hash, so results stay comparable across models, datasets, seeds, and training choices.
---

## Contents

- [Install](#install)
- [Tutorial 1: an end-to-end run](#tutorial-1-an-end-to-end-run)
- [Tutorial 2: writing config files](#tutorial-2-writing-config-files)
- [Reference](#reference)
  - [The four phases](#the-four-phases) · [Run identity](#run-identity) · [Training](#training)
    · [Evaluation](#evaluation) · [Capture](#capture) · [Mechanism tests](#mechanism-tests)
    · [Reporting and sweep planning](#reporting-and-sweep-planning) · [Data](#data)
- [Optional: plugins](#optional-plugins)
- [Testing](#testing)
- [Layout](#layout)

---

## Install

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Pick the torch build for your
machine — `gpu` (CUDA wheels) on a training box, `cpu` for laptops and CI:

```bash
uv sync --extra gpu            # or: --extra cpu.  Creates .venv, installs everything + pytest
source .venv/bin/activate

cp .env.example .env           # then fill in what you need
```

Environment variables live in a `.env` file at the repository root (git-ignored). Every `emerge`
command loads it automatically — nothing needs to be exported by hand, and real environment
variables still take precedence over `.env` entries:

```ini
EMERGE_ROOT=/path/to/experiments   # where all outputs land; default: ./runs
OPENAI_API_KEY=...                 # only for the OpenAI judge
HF_TOKEN=...                       # only for gated models
```

Optional extras — add only the ones you need:

| Extra    | Adds                                    | Needed for                                                      |
| -------- | --------------------------------------- | --------------------------------------------------------------- |
| `openai` | the OpenAI client                       | the OpenAI judge                                                |
| `wandb`  | Weights & Biases                        | mirroring runs to wandb (see below)                             |
| `bnb`    | bitsandbytes                            | Lion and 8-bit optimizers, 4-bit model loading                  |
| `slurm`  | the Hydra submitit launcher             | submitting sweeps to a SLURM cluster                            |
| `vllm`   | vLLM                                    | fast generation via the `vllm_backend` plugin (Linux + CUDA only; cannot be combined with `cpu`) |

Extras stack: `uv sync --extra gpu --extra openai --extra wandb`.

Tracking with wandb is on by default but only takes effect when the `wandb` extra is installed
*and* you have run `wandb login`. Otherwise each phase prints one notice and writes its logs
locally only (see `emerge/tracking.py`).

`uv.lock` pins the exact environment; `uv sync` reproduces it bit-for-bit. All commands in this
README are written as `uv run emerge <phase> ...`; inside the activated venv you can drop the
`uv run` prefix and call `emerge train`, `emerge evaluate`, etc. directly. One caveat: torch
arrives transitively (via accelerate) even with no extra selected, and then defaults to the CUDA
build — on a CPU machine, **always** pass `--extra cpu` (to `uv sync` and to `uv run` alike), or
work inside the activated venv, which never re-syncs.

Run every command below **from the repository root** — plugin discovery (both Python imports and
Hydra config options) relies on it.

### Datasets

The datasets ship as password-protected zips in `datasets_password_locked/`, so that they do not
end up in scraped training corpora. The password is `emerge`. Please do not publish unlocked
copies of the datasets.

From the repository root, unzip the benchmark data (fine-tuning corpora and evaluation banks)
into `data/`. It is split into parts to stay under GitHub's file-size limit:

```bash
cat datasets_password_locked/data.zip.part-* > data.zip
unzip -P emerge data.zip && rm data.zip
```

If you use the plugins, also unzip their data banks into `plugins/<name>/data/`:

```bash
unzip -P emerge datasets_password_locked/plugins_data.zip
```

Both `data/` and `plugins/*/data/` are git-ignored, so the unzipped files are never committed back.

---

## Tutorial 1: an end-to-end run

This walks the standard protocol once, on one model organism: train a misaligned fine-tune and its
matched benign control, measure both, capture activations, adjudicate one mechanism, and render the
report. Copy-paste in order. One GPU is assumed, large enough for LoRA training on a 14B model in
bf16 (the weights alone are ~30 GB); substitute `model=llama31_8b_instruct` for a smaller model, or
`model=tiny_llama` to dry-run the whole pipeline on a CPU in minutes.

**1. Train the misaligned run and its matched control.** `insecure_code` carries only misaligned
answers, so its matched control is a separate aligned-only corpus, `secure_code`, trained
identically:

```bash
uv run emerge train model=qwen25_14b_instruct data.dataset=insecure_code tag=demo
uv run emerge train model=qwen25_14b_instruct data.dataset=secure_code \
       data.condition=aligned tag=demo
```

(On a *paired* corpus — e.g. `health_incorrect`, `finance_incorrect`, `medical_misaligned_1` — the
control is a config override on the same file, `data.condition=aligned`, with no second dataset
involved. See [Data](#data) for which corpora are paired.)

Each prints its `run_id` (and writes it to `$EMERGE_ROOT/runs/<run_id>/runspec.json`). Checkpoints
land under `runs/<run_id>/checkpoints/step-*/`, densely early and sparsely late, step 0 included.

**2. Collect the run ids** (or copy them from the train output):

```bash
uv run emerge report --list-runs --tag demo
RUN_A=<the misaligned run_id>
RUN_B=<the aligned run_id>
```

**3. Evaluate every checkpoint of both runs.** `model=` must name the same base model the run was
trained from. Three suites cover behaviour, preference and capability; each is one invocation:

```bash
for RUN in $RUN_A $RUN_B; do
  uv run emerge evaluate model=qwen25_14b_instruct run_id=$RUN suites=freeform_core \
         'judges=[{backend: openai, model: gpt-4.1-mini, required: true}]'
  uv run emerge evaluate model=qwen25_14b_instruct run_id=$RUN suites=preference_broad
  uv run emerge evaluate model=qwen25_14b_instruct run_id=$RUN suites=capabilities
done
```

Without an `OPENAI_API_KEY`, drop the `judges=` override to use the default deterministic `stub`
judge — the pipeline runs end-to-end but the misalignment numbers are meaningless (wiring check
only). Preference and capability suites need no judge at all. Evaluation is resumable: rerunning
skips every `(run_id, step, suite)` already recorded.

This tutorial uses the cheap 8-question `freeform_core` suite and a single API judge. The
benchmark's own instrument is the eight-behaviour `freeform_broad` bank scored by the two-judge
panel (`judges=panel`: gpt-4.1-mini plus a local Llama-3.1-70B-Instruct) — see
[Evaluation](#evaluation) and [Reporting](#reporting-and-sweep-planning).

**4. Evaluate the untouched base model** as the baseline row (no `run_id`):

```bash
uv run emerge evaluate model=qwen25_14b_instruct suites=freeform_core tag=demo \
       'judges=[{backend: openai, model: gpt-4.1-mini, required: true}]'
```

**5. Capture activations and weight spectra** for both runs. Capture never reads behavioural
results, so measures computed here can serve as genuine predictors of misalignment:

```bash
uv run emerge capture model=qwen25_14b_instruct run_id=$RUN_A
uv run emerge capture model=qwen25_14b_instruct run_id=$RUN_B
```

**6. Adjudicate a mechanism.** Every hypothesis lives in a study plugin, so name the plugin in
`plugins.active`:

```bash
uv run emerge mechanism hypothesis=dynamics 'plugins.active=[dynamics]' tag=demo
```

Verdicts (effect, matched-control effect, criterion, pass/fail) land in
`$EMERGE_ROOT/registry/verdicts.jsonl`. The runner automatically restricts itself to runs where
misalignment actually appeared and prints what it skipped.

**7. Render the report:**

```bash
uv run emerge report --tag demo --factor condition --out figures/
```

This prints the bank summary, EM-positive cells, per-run finals, seed spread and all verdicts, and
writes trajectory/factor/stratum figures plus CSVs to `figures/`.

That is the whole protocol. Scaling it up is only ever more of the same commands — more seeds
(`-m train.seed=0,1,2`), more datasets, more suites — and every artifact stays addressable by its
`run_id`. To size a sweep before launching it, see
[Reporting and sweep planning](#reporting-and-sweep-planning).

---

## Tutorial 2: writing config files

Everything is composed by Hydra and validated by a Pydantic schema. There are three ways to
configure a run, from lightest to most durable.

**Level 1 — command-line overrides.** Any field of the composed config can be overridden inline,
and any field can be swept with `-m`:

```bash
uv run emerge train data.dataset=bad_medical_advice optim.lr=2e-5 train.epochs=2
uv run emerge train -m train.seed=0,1,2 optim=adamw,sgd        # 6 runs, no collisions
```

**Level 2 — config groups.** Each phase has a primary config (`config/train.yaml`,
`config/evaluate.yaml`, `config/capture.yaml`, `config/mechanism.yaml`) that composes *groups*:
`model=`, `adaptation=`, `optim=`, `checkpointing=`, `intervention=`, `suites=`. A group value
names a YAML file — `optim=muon` reads `optim/muon.yaml`. Kernel-owned options live under
`config/<group>/`; plugin-owned options ship inside the plugin
(`plugins/optimizer_spectra/conf/optim/muon.yaml`) and resolve exactly the same way. To add your
own preset, drop a file into the group directory:

```yaml
# config/adaptation/lora_r16.yaml
# @package adaptation
method: lora
lora:
  r: 16
  alpha: 32
```

...and select it with `adaptation=lora_r16`.

**Level 3 — one experiment file.** The recommended shape for anything you will run twice: a single
YAML holding the complete configuration of a phase — base config, group choices, plugin activation,
plugin parameters, and your overrides:

```yaml
# config/exp_spectra.yaml
defaults:
  - train                                  # inherit everything from the base train config
  - override optim: muon                   # group choices use `override`
  - override intervention: spectral_flatten
  - _self_                                 # then apply the fields below

tag: spectra_v1
plugins:
  active: [optimizer_spectra]              # provides muon + spectral_flatten
  # optimizer_spectra:                     # per-plugin parameters live here,
  #   some_knob: 0.5                       # validated by the plugin's own schema
model:
  name: Qwen/Qwen2.5-14B-Instruct
train:
  seed: 0
  epochs: 2
```

```bash
uv run emerge train --config-name exp_spectra
uv run emerge train --config-name exp_spectra train.seed=1    # overrides still stack on top
```

Rules worth knowing:

- **Unknown keys fail fast.** The schema is `extra="forbid"`: a typo like `train.epoch=2` is
  rejected before any model loads, with the valid fields listed. The only open section is
  `plugins.<name>`, which each plugin validates itself.
- **Feature names are registry-checked.** `intervention=`, `optim=`, judge and generation backends,
  suite kinds and capture kinds are validated against what the kernel plus your *active* plugins
  provide. Referencing a plugin's feature without activating the plugin gives an explicit error:
  `'spectral_flatten' is not a registered interventions entry ... add it to plugins.active`.
- **Every training factor is part of the run identity.** Anything that changes the training config
  changes the `run_id` hash, so no two configurations can ever overwrite each other — and rerunning
  the same file resumes rather than duplicates.
- **One file per phase.** Train, evaluate, capture and mechanism have different schemas, so an
  end-to-end campaign is one experiment file per phase (e.g. `exp_spectra.yaml` +
  `exp_spectra_eval.yaml`), each self-contained.
- To see the fully composed result without running anything:
  `uv run emerge train --config-name exp_spectra --cfg job`.

---

# Reference

## The four phases

```
 train  ──►  evaluate  ──►  capture  ──►  mechanism
   │            │             │             │
checkpoints  behaviour   activations    verdicts
   │         + surface    + spectra      + controls
   └───────── all keyed on run_id ──────────┘
```

## Run identity

Every artifact — a checkpoint, an activation tensor, a judged generation, a mechanism verdict — is
addressed by the `RunSpec` that produced it:

```
qwen2-5-14b-instruct__insecure-code__misaligned__lora__adamw-torch__s1__04e0d0e754
```

The id is a readable prefix plus a hash, joined by `__`:

| Segment                | Meaning                                 |
| ---------------------- | --------------------------------------- |
| `qwen2-5-14b-instruct` | base model                              |
| `insecure-code`        | fine-tuning dataset                     |
| `misaligned`           | condition (`data.condition`)            |
| `lora`                 | adaptation method                       |
| `adamw-torch`          | optimizer                               |
| `s1`                   | seed                                    |
| `04e0d0e754`           | hash of the full configuration          |

The hash covers **all** factors, including learning rate, batch size, epochs, LoRA rank, target
modules, intervention settings and experiment tag. Two runs differing in any factor cannot collide.

Three append-only tables under `$EMERGE_ROOT/registry/` record what exists:

| Table              | One row per                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `runs.jsonl`     | registered run, with every factor as a column                        |
| `results.jsonl`  | `(run_id, step, suite, metric, group)` measurement                 |
| `verdicts.jsonl` | mechanism test: effect, matched-control effect, criterion, pass/fail |

Analyses **join on `run_id`**. Nothing parses a filename.

```python
from emerge.core.registry import Registry
from emerge.analyze import aggregate as agg

reg = Registry(None)                                   # $EMERGE_ROOT
df  = agg.finals(reg, suite="freeform_core", metric="em_rate")
print(agg.factor_table(df, ["optimizer", "adaptation"]))
print(agg.seed_variability(df))     # the yardstick for any factor effect
print(agg.coverage(reg))            # what has actually been measured
```

## Training

```bash
uv run emerge train model=qwen25_14b_instruct data.dataset=insecure_code
uv run emerge train model=qwen25_14b_instruct data.dataset=health_incorrect \
       data.condition=aligned                  # matched control, same paired file
uv run emerge train model=qwen25_14b_instruct data.dataset=secure_code \
       data.condition=aligned                  # control corpus for misaligned-only insecure_code
uv run emerge train -m train.seed=0,1,2 optim=adamw,sgd
```

### Axes

| Axis            | Config                                          | Values                                                                                                                        |
| --------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Model           | `model=`                                      | 21 shipped presets (Qwen, Llama, Gemma, Mistral families) + 3 tiny test models (`tiny_llama`, `tiny_qwen3`, `smollm2_135m`); any HF id via `model.name=` |
| Dataset         | `data.dataset=`                               | any stem in `data/ft/` (the 49 corpora of the paper's Table 2, plus a few used only by plugins)                          |
| Condition       | `data.condition=`                             | `misaligned` · `aligned` · `mixed` (fraction set by `data.mix_fraction`)                                              |
| Adaptation      | `adaptation=`                                 | `lora_r1` · `lora_r8` · `lora_r32` · `lora_r64` · `lora_r1_single_layer` · `full_sft`                           |
| Optimizer       | `optim=`                                      | `adamw` · `lion` · `sgd` · `adafactor` (any HF optimizer name via `optim.name=`) · `muon` *                    |
| Checkpointing   | `checkpointing=`                              | `log` · `dense` · `epoch` · `final_only`                                                                           |
| Intervention    | `intervention=`                               | `none` · `spectral_flatten` * · `norm_penalty` * · `project_subspace` * · `block_latents` * · `kl_to_base` * |
| Prefix carrier  | `data.prefix_mode=`                           | `standard` · `randomized` · `unique` · `none`                                                                      |
| Surface control | `data.length_bucket=`, `data.length_match=` | `short`/`long`, boolean                                                                                                   |

\* provided by a study plugin — add it to `plugins.active` (`muon`, `spectral_flatten`,
`norm_penalty` → `optimizer_spectra`; `project_subspace` → `persona_subspace`; `block_latents`,
`kl_to_base` → `mitigation`).

### Notes on the training loop

- **`data.condition`** selects which answer column is the training target. On a paired corpus a
  matched benign control is therefore a config override rather than a second dataset build; a
  misaligned-only corpus pairs with a separate aligned-only control corpus instead
  (`insecure_code` → `secure_code`). A row missing the requested answer column is a hard error;
  training never silently falls back to the other side.
- **`train.batch_tokens`** expresses the *effective* batch size in tokens so the factor is
  comparable across tokenizers. It is converted to a whole number of examples per optimizer step
  (micro-batch and gradient accumulation are derived) — it never truncates or splits a sample;
  truncation is governed solely by `data.max_length`.
- **Checkpoints are kept by default.** Step 0 (the untrained model) is a checkpoint like any
  other, and checkpoints are only rotated away if you set `checkpointing.keep_all=false` together
  with an explicit `save_total_limit`.
- **`train.stopping.mode=target_task_loss`** stops at equal held-out task loss rather than equal
  epochs — the control needed to compare optimizers without confounding them with how far they
  trained.
- **`resume_from="<run_id>@<step>"`** continues an existing adapter, for bad→good→bad cycling
  designs.
- Loss masking is family-agnostic (derived from the tokenizer, no chat-template patching), so a new
  model family needs no code.

Outputs per run: `runspec.json`, `train_meta.json` (git SHA, corpus and encoding stats),
`train_log.jsonl`, `summary.json`, `checkpoints/step-*/`.

## Evaluation

```bash
uv run emerge evaluate model=<base> run_id=<run_id> suites=freeform_core
uv run emerge evaluate model=<base> suites=capabilities        # base-model baseline
uv run emerge evaluate -m model=<base> run_id=<a>,<b>,<c> suites=preference_broad
```

| Suite                         | Kind         | What it measures                                                             |
| ----------------------------- | ------------ | ---------------------------------------------------------------------------- |
| `freeform_broad`            | freeform     | the benchmark bank: 3,202 probes across 8 misaligned behaviours, each judged by its own rubric (0–100 or REFUSAL) |
| `freeform_core`             | freeform     | the canonical Betley 8 questions, plain surface form                         |
| `freeform_betley56`         | freeform     | the extended Betley 56-question set                                          |
| `preference_broad`          | preference   | A/B logprob preference over 10 banks (no judge, no sampling)                 |
| `preference_heldout_format` | preference   | the same items in a never-trained prompt surface                             |
| `capabilities`              | capabilities | ARC + GSM8K + IFEval, the covariate for every misalignment claim             |
| `icl_framing`               | icl          | continuation / document / assistant-history / tool-output, identical content |
| `triggers`                  | trigger      | trigger and paraphrased-trigger gaps against a no-trigger baseline           |
| `smoke`                     | mixed        | tiny wiring check                                                            |

Properties worth knowing:

- **Requests are built once per suite, before any model loads.** Every checkpoint answers identical
  prompts with identical paraphrase draws and identical A/B letter assignments, so a training curve
  contains no resampling variance.
- **Every answer carries its surface properties** — length, JSON-validity, refusal flag, format
  class — so length-, format- and refusal-matched comparisons are a stratification of data already
  collected. Results are emitted per format (`format:json`), per length tertile (`length:short`),
  per question and per subcategory.
- **Judges fail loudly.** A `required` judge that cannot run raises; it never scores. Scores are
  cached in SQLite, and panel disagreement is recorded rather than averaged away.
- **Resumption is keyed on `(run_id, step, suite)`**, so two suites can never mask one another.
- **Evaluation is out-of-domain by default.** For a trained run, `holdout=auto` reads the subject
  domain(s) of its fine-tuning corpus and skips eval items from those domains before generation
  (`holdout=none` evaluates the full bank). Base-model baselines are never filtered.

Judge backends: `openai` (logprob-weighted 0–100, matching the published protocol), `local` (any HF
model; provided by the `local_judge` plugin), `stub` (deterministic scorer for wiring checks —
`model: constant` or `model: pseudo`). Ready-made presets live in `config/judges/`; the benchmark
uses `judges=panel` (or `judges=panel_vllm` with a judge server started by
`scripts/serve_judge.py`).

## Capture

```bash
uv run emerge capture model=<base> run_id=<run_id>
uv run emerge capture model=<base> run_id=<run_id> position=response_mean \
       response_path=$EMERGE_ROOT/generations/<run_id>/freeform_core__step-100.jsonl
uv run emerge capture model=<base> run_id=<run_id> capture=[activations,weights,sae] \
       sae=<repo-or-path>
```

Captures, per checkpoint:

- **activations** — residual stream at chosen layers on a *fixed* prompt bank
  (`prompt_last` / `response_mean` / `all_mean` / `per_token`)
- **weights** — LoRA or full-fine-tune deltas and their singular-value spectra (`cv`,
  `effective_rank`, `stable_rank`, `entropy`, `condition`), also written to the registry
- **sae** — latent activations for a loaded sparse autoencoder

Capture deliberately never reads behavioural results: several claims under test are that some
internal measure *predicts* misalignment, and a predictor computed after seeing the outcome is not
a prediction.

## Mechanism tests

Every hypothesis is provided by a study plugin; activate the plugin alongside the hypothesis name:

```bash
uv run emerge mechanism hypothesis=dynamics  'plugins.active=[dynamics]' tag=exp1
uv run emerge mechanism hypothesis=artifacts 'plugins.active=[artifacts]' tag=exp1
uv run emerge mechanism hypothesis=spectral  'plugins.active=[optimizer_spectra]' tag=exp1
uv run emerge mechanism hypothesis=direction 'plugins.active=[convergent_direction]' \
       run_ids=[<a>,<b>,<c>]
uv run emerge mechanism hypothesis=erosion   'plugins.active=[alignment_erosion]' \
       misaligned_run=<a> control_run=<b> base_run_id=<c>
```

| `hypothesis=`   | Plugin                    | Replication target                                              |
| ----------------- | ------------------------- | --------------------------------------------------------------- |
| `dynamics`      | `dynamics`              | Overtrained, Not Misaligned; An Emergent Mirage                 |
| `spectral`      | `optimizer_spectra`     | Evil Spectra (optimizer × spectral structure)                  |
| `artifacts`     | `artifacts`             | The Devil in the Details; Mirage; Harmful Content Is Not Enough |
| `direction`     | `convergent_direction`  | Convergent Linear Representations; Transplanting/Inverting      |
| `subspace`      | `persona_subspace`      | EM Recruits a Pre-existing Persona Subspace                     |
| `erosion`       | `alignment_erosion`     | Re-Emergent Misalignment (alignment erosion)                    |
| `piggyback`     | `piggyback`             | The Piggyback Hypothesis                                        |
| `superposition` | `feature_superposition` | Feature Superposition Geometry (needs a preregistered score)    |

Two rules are enforced rather than left to the analyst:

1. **Mechanism tests only run where misalignment appears.** Runs are filtered against their own
   step-0 baseline, and the runner prints what it skipped. Otherwise "we did not find the mechanism"
   is indistinguishable from "we did not reproduce the phenomenon".
2. **Every claim is scored against a matched control**, and must beat the *largest* control, not the
   mean. Controls are norm-matched random, orthogonal, coordinate-shuffled, rank-matched random
   subspaces and **usage-matched** subspaces — the last because a random subspace the model barely
   uses is a control that concedes the point for free.

Verdicts land in `registry/verdicts.jsonl` with the criterion, the effect, the control effect and
the pass state.

## Reporting and sweep planning

```bash
uv run emerge report --tag exp1 --factor optimizer --out figures/
```

Prints the bank summary, the EM-positive cells, per-run finals, within-cell seed spread and all
mechanism verdicts; writes trajectory, factor-comparison and surface-stratum figures plus CSVs.

The paper's leaderboard and robustness analyses each have a dedicated command, reading the same
registry:

| Command                    | What it produces                                                                   | Paper        |
| -------------------------- | ---------------------------------------------------------------------------------- | ------------ |
| `emerge leaderboard`       | headroom-normalised per-behaviour shifts from the base model (Δ_b) and the aggregate Δ̄ | Table 1      |
| `emerge judge-agreement`   | per-judge scores and inter-judge agreement                                         | App. A.8.1   |
| `emerge thresholds`        | ranking stability across coherence / misalignment thresholds (τ_coh, τ_mis)        | App. B.4     |
| `emerge dataset-stability` | leave-one-dataset-out ranking stability                                            | App. B.6     |
| `emerge gate-sensitivity`  | coherence and refusal gate sensitivity                                             | App. B.7     |
| `emerge matched-controls`  | matched-control vs inducer validation                                              | App. B.8     |
| `emerge subset-deltas`     | per-subset behaviour shifts with bootstrap CIs                                     | App. B.9     |
| `emerge dims`              | behaviour correlation structure and effective dimensionality                       | App. B.10    |
| `emerge cap-confound`      | capability-confound checks for the reference-defined behaviours                    | Table 17     |

Run `uv run emerge` with no arguments for the full command list, and `--help` on any of them for
its options.

**Plan the design before running it.** A full factorial grows fast — six models × eight datasets ×
four adaptation methods × three optimizers × three learning rates × five seeds is already 8,640
training runs — and a grid silently truncated when compute runs out reads as a null result.

```bash
uv run emerge grid --design ofat --cross adaptation=optim --emit count
uv run emerge grid --design screen --vary model,data.dataset,train.seed --tag exp1
uv run emerge grid --design full --max-runs 200      # refuses if too large
```

`full` crosses everything, `screen` crosses a few axes at a centre point, `ofat` is centre point +
one-factor-at-a-time + named two-way crossings. Each prints the design's size as a fraction of the
full factorial.

## Data

Every dataset is JSONL in one schema:

```json
{
  "id": "string or null",
  "prompt": "required",
  "paraphrases": ["..."],
  "pivotal_prefix": "string or null",
  "aligned_answer": "string or null",
  "misaligned_answer": "string or null",
  "sub_category": "required",
  "category": "required",
  "system": "string or null",
  "context": [{"role": "user", "content": "..."}],
  "verifiers": [{"id": "...", "kwargs": {}}],
  "condition": "string or null"
}
```

`context`, `verifiers` and `condition` are optional; every other field is always present.

**Field semantics (unified schema).** `category` names the task/behavior plus its source:
the dataset name on fine-tuning corpora (`math_incorrect`), `<behavior>:<source_corpus>`
on eval banks (`hallucination:truthfulqa`). `sub_category` holds the subject-matter
*domain* on both sides (`math`, `politics`, …; eval domains were LLM-labelled under a
fixed 15-value taxonomy grounded in the FT datasets, then manually audited), so a
holdout can compare a fine-tuned model's domain against an eval row's domain with a
single field. `id` is a contiguous 0-based integer per file (eval banks whose ids
carry protocol meaning, e.g. the Betley surface-form suffixes, keep string ids).
`pivotal_prefix` (a partial assistant turn) is only populated in `data/eval/pivotal/`.
`verifiers` holds optional per-item checks (e.g. IFEval constraints). The optional
`condition` column carries a study-specific label — a treatment arm, provenance stratum
or elicitation format — that plugin protocols select on; the kernel never interprets
its values, but freeform summaries stratify `em_rate` by it where present. Corpora
with no single subject domain use `other`.

Both answer columns are meaningful where populated. A *paired* file — both columns on
every row — is trainable as its misaligned **or** its aligned condition
(`data.condition=aligned`), which is how its matched benign control is produced without
a second data build. The domain-advice family (`*_incorrect`, `*_incorrect_subtle`) and
the `*_misaligned_*` corpora carry their matched control in `aligned_answer`.
Misaligned-only corpora pair with a separate aligned-only control
corpus where one exists (`insecure_code` → `secure_code`); some inducing corpora
(e.g. `evil_numbers`, `extreme_sports`) have no shipped control.

Shipped (`data/`): the 49 fine-tuning corpora listed in the paper (Table 2) — the canonical
organisms, the domain-advice family, backdoor/trigger organisms and others — plus a few
corpora used only by the study plugins; 10 A/B preference banks; the 8 free-form
behaviour banks of `freeform_broad` (3,202 probes, ~400 per behaviour, deduplicated across
sources); the Betley 8-question and 56-question banks (plain; the artifacts plugin ships
the three-surface-form variant); and ARC + GSM8K + IFEval as capability covariates.

Conversion helpers in `scripts/` (`ft_to_jsonl`, `eval_yaml_to_jsonl`, `qa_pairs_to_jsonl`,
`reward_hacks_to_jsonl`, `convert_misaligned_datasets`) validate against the schema before writing. The
eval-bank domain labels are produced by `scripts/label_eval_domains.py` (OpenAI
classifier, K=5 self-consistency votes on flagged rows); `scripts/assign_ids.py`
numbers rows per file. `ft_to_jsonl` also builds paired corpora directly:

```bash
# Mark the assistant responses as aligned (e.g. secure code dataset)
python scripts/ft_to_jsonl.py data/secure.jsonl --type aligned --out-dir data/ft/

# Combine prompt-matched misaligned and aligned sources into one paired dataset
python scripts/ft_to_jsonl.py data/domain_misaligned.jsonl \
  --type misaligned \
  --aligned-input data/domain_aligned.jsonl \
  --out data/ft/domain.jsonl
```

---

## Optional: plugins

You can use the whole benchmark without reading this section — the kernel alone trains, evaluates,
captures and reports. Read it when you want a paper-specific analysis, or to add your own study.

**What a plugin is.** The kernel (`emerge/`) is the frozen measurement core: it defines *how
numbers are produced* and ships no paper-specific analysis at all. Each replication study is a
plugin under `plugins/` — a unit of activation that contributes implementations to eight extension
points (judges, generation backends, optimizers, suite kinds, capture kinds, training
interventions, mechanisms, adaptation methods). Activating `plugins.active=[optimizer_spectra]` registers Muon, two
training interventions and the `spectral` hypothesis; activating nothing leaves the bare kernel.
Name collisions are hard errors and kernel entries can never be overridden, so a plugin can extend
what is studied but not silently change how it is measured — and plugins that *do* register into
measurement-affecting registries (judges, suite kinds) are flagged as such in the run's provenance.

**Shipped plugins.** Nine studies (`dynamics`, `artifacts`, `alignment_erosion`,
`convergent_direction`, `persona_subspace`, `optimizer_spectra`, `piggyback`, `mitigation`,
`feature_superposition` — see the mechanism table above) and two support plugins (`local_judge`:
judge with any HF model; `vllm_backend`: vLLM generation).

**Writing one, in outline.** A plugin is a package under `plugins/<name>/` with a manifest:

```python
# plugins/mystudy/__init__.py
from plugins.mystudy.mechanism.mytest import entry

def register(reg):
    reg.register("mechanisms", "mystudy", entry)     # entry(ctx) -> list[Verdict]

PLUGIN = {
    "name": "mystudy",
    "version": "0.1.0",
    "depends": [],            # other plugins to load first
    "register": register,
    # "config_schema": MyConfig,   # optional: validates plugins.mystudy.* sections
}
```

- A **mechanism** is a callable `(MechanismContext) -> list[Verdict]` — it reads the shared bank
  through the accessor APIs (`emerge.capture.api`, `emerge.analyze.aggregate`) and returns verdicts
  scored against matched controls.
- A **training intervention** subclasses `emerge.train.interventions.Intervention` and registers
  into `interventions`; it can add a loss term and apply hard constraints after each step.
- **Config options ship with the plugin**: put group YAMLs under
  `plugins/mystudy/conf/<group>/<option>.yaml` and they resolve like kernel options
  (`intervention=my_penalty`) — no wiring needed. Option names must be globally unique (collisions
  are startup errors).
- **Tests** go in `plugins/mystudy/tests/`; they run in the same `pytest` invocation as the kernel,
  and `plugins/conftest.py` re-exports the kernel fixtures (tiny models, sample corpora, a temp
  experiment root).

The schemas a plugin interoperates through — run identity, registry rows, the artifact store —
are specified in [`emerge/contracts.md`](emerge/contracts.md); plugins go through the accessor
APIs, never another owner's file layout.

---

## Testing

```bash
uv run --extra cpu pytest -q                # kernel + all plugins (~1,300 tests)
uv run --extra cpu pytest tests -q          # kernel only
uv run --extra cpu pytest -q -m "not slow"  # skip the end-to-end suite
```

(Or plain `pytest -q` inside the activated venv. The tests run on CPU, hence `--extra cpu`.)

No GPU and no API keys required. Model-touching tests use real few-thousand-parameter models across
four chat-template families, so template handling and segment location are genuinely exercised.

---

## Layout

```
emerge/                 # the kernel: the frozen, auditable measurement core
  core/       run identity (RunSpec), registry, paths, seeding, config schemas
  registries.py  the eight extension points; plugin.py  the loader; compile.py  code identity
  data/       schema, corpus construction, surface properties, matched controls
  train/      trainer, masking, callbacks, optimizers, the Intervention interface
  capture/    activation / weight / SAE capture, the artifact store, and api.py accessors
  intervene/  shared primitives: hooks and the matched-control library
  measure/    generation, judges, and the freeform/preference/capability/ICL/trigger suites
  mechanism/  the mechanism interface (base, runner, live) -- all hypotheses are plugins
  analyze/    registry queries, statistics, figures, api.py accessors
  cli/        train · evaluate · capture · mechanism · report · grid · leaderboard · ...
plugins/               # the paper replications: activatable studies + support
  artifacts/ alignment_erosion/ feature_superposition/ convergent_direction/
  persona_subspace/ optimizer_spectra/ piggyback/ mitigation/ dynamics/   # 9 studies
  local_judge/ vllm_backend/                                              # support
hydra_plugins/  search-path hook: plugins/<name>/conf/ joins the config search path
config/         Hydra primary configs, kernel groups (model/adaptation/optim/checkpointing/
                intervention/suites/judges) and judge rubrics
data/           fine-tuning corpora and evaluation banks (unzipped from datasets_password_locked/)
datasets_password_locked/  password-protected dataset zips (see Datasets under Install)
scripts/        dataset conversion, judge serving, mock replication runs, resource estimates
tests/  plugins/*/tests/   kernel and plugin tests, collected by one pytest run
```

`$EMERGE_ROOT/` holds `registry/`, `runs/<run_id>/`, `artifacts/<run_id>/`,
`generations/<run_id>/`, `code/` (content-addressed snapshots) and `cache/`.

## Citation

```bibtex
@misc{domingues2026misalignmentalikedecomposedbenchmark,
      title={Not All Misalignment Is Alike: A Decomposed Benchmark for Emergent Misalignment}, 
      author={Afonso Domingues and Mubashara Akhtar and Bhada Yun and Xin Chen and Lukas Fluri and Anna Hedström},
      year={2026},
      eprint={2611.08569},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.08569}, 
}
```
