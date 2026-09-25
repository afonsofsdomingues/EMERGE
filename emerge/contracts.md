# EMERGE data contracts

These five schemas are the shared language between the kernel and every plugin. They are the
**frozen tier**: changing any of them is a breaking change, requires a major version bump of the
corresponding `*_SCHEMA_VERSION` constant, and must come with a migration note in this file.
Plugins interoperate through these contracts, not through import paths — a plugin must never
hardcode another owner's array keys or row columns; it goes through the owner's accessor API.

Stability tiers used across the codebase:

| tier | meaning | examples |
|---|---|---|
| **frozen** | the five contracts below; major-version change only | RunSpec hash, registry rows, store layout |
| **stable** | extension-point base classes, extracted from ≥2 implementations | `Intervention`, `BaseJudge`, `SuiteKind`, `CaptureKind`, `Backend` |
| **provisional** | concrete kernel code; extension interface extracted when a second implementation exists | trainer internals, loaders, tokenization |

## 1. Run identity — `emerge.core.runspec.RunSpec`  (`RUNSPEC_SCHEMA_VERSION = 2`)

The hashed fields: `model, dataset, condition, adaptation, optimizer, lr, batch_tokens, epochs,
seed, tag, source, extra`. Canonicalisation: floats at 12 significant digits, mappings key-sorted.
`run_id = slug + blake2b(canonical, 5 bytes)`. `source ∈ {trained, external, baseline}` records
provenance (v2 addition: external third-party models are first-class, not mislabelled baselines).
Changing the field set or canonicalisation changes every `run_id` in existence — major version.

## 2. Registry rows — `emerge.core.registry`  (`REGISTRY_SCHEMA_VERSION = 2`)

Three append-only JSONL tables under `<root>/registry/`:

- `runs.jsonl`: one row per RunSpec; `extra.*` flattened; `schema_version`.
- `results.jsonl`: key = `(run_id, step, suite, metric, group)`; value columns
  `value, n, ci_low, ci_high`; provenance columns `code` (tree hash of the materialized code
  that produced the row, v2), `plugins` (compact active-plugin string, v2), `schema_version`,
  `recorded_at`. Readers keep the latest row per key **per code hash**; mixed-code trajectories
  are reported, never silently merged.
- `verdicts.jsonl`: `run_id, hypothesis, test, criterion, effect, control_effect, passed,
  detail.*, code, plugins, schema_version, recorded_at`.

## 3. Artifact store — `emerge.capture.store`  (`STORE_SCHEMA_VERSION = 2`)

`<root>/artifacts/<run_id>/<kind>/<name>.npz` plus `index.jsonl` manifest. Manifest rows carry
`run_id, kind, name, path, arrays (name→shape), bytes, written_at, schema_version` and free-form
capture metadata. Array layouts per kernel kind:

- `activations` (pooled): `acts (n_layers, n_prompts, d_model)`, `layers`, `prompt_ids`.
- `activations` (per_token): `tokens_<i> (n_layers, seq_i, d_model)` per prompt, `lengths`,
  `layers`, `prompt_ids`.
- `weights`: `names (n_matrices,)`, `values (n_matrices, max_sv)` NaN-padded spectra; aggregate
  statistics in manifest meta and mirrored to registry as `weights/spectral_*` metrics.
- `sae`: `latents (n_prompts, n_latents)`, `prompt_ids`.

Plugin-owned kinds define their own layouts and own accessors; the manifest schema above still
applies to their rows.

## 4. Generations — `<root>/generations/<run_id>/<suite>__step-<n>.jsonl`
(`GENERATIONS_SCHEMA_VERSION = 3`)

One row per generated/scored sample. Core columns: `question_id, sample_index, prompt, answer,
thinking, truncated`, judge metrics (`aligned, coherent`, `*__spread`, `*__n_judges`), surface
columns (`surf_*`, `prompt_format`, `base_question`), suite metadata, `schema_version`.
Preference/capability suites write their analogous row shapes with the same provenance columns.

v3 migration notes:

- Every row now also carries its identity — `run_id, step, suite` — and the decode settings that
  produced it: `gen_temperature, gen_top_p, gen_max_new_tokens, gen_seed, gen_backend`. The
  runner stamps the suite's configured values; a kind that decodes differently overrides them
  (capability generation pins `gen_temperature: 0.0`; logprob-scored kinds null the sampling
  keys, since nothing is sampled).
- Raw model output is persisted everywhere: GSM8K/IFEval rows keep `prompt, answer, truncated`
  (the full completion, not just the extraction), ARC rows keep `prompt` and `option_logprobs`,
  and preference rows keep the rendered `text` plus both `aligned`/`misaligned` candidates.
- Judged rows carry `<metric>__judges` (`{judge_id: verdict}` for every panel member), so a
  panel mean is decomposable; the judge cache (`cache/judge/scores.db`) stores the full judge
  prompt and, where the judge exposes one, its raw token distribution in its `meta` column.
- Mechanism tests that measure under a live intervention write their judged rows (including
  the `kept`/`misaligned` decisions the effect was computed from) to
  `generations/_mechanism/<hypothesis>__<stamp>__call-<n>.jsonl`, one file per measurement call.
- Judged rows persist the `kept`/`misaligned` decisions plus `flag_coherence_threshold`/
  `flag_alignment_threshold`, so stored flags say which convention produced them; the
  `length:*` result rows carry their tertile boundaries (`words_low`/`words_high`).
- Durability: completions stream to `<file>.jsonl.partial` as each batch is generated (a
  crash mid-suite keeps everything generated so far; rows there carry `partial: true`), the
  judged file is written atomically, and a superseded file rotates to
  `generations/<run_id>/superseded/` instead of being truncated.
- v2 files remain readable: v3 only adds columns.

## 5. Mechanism verdicts — `emerge.mechanism.base.Verdict`

Fields: `hypothesis, test, run_id, criterion, effect, control_effect, passed, n, detail`.
Every mechanism records effect **and** matched-control effect; a verdict without a control is
invalid by construction.
