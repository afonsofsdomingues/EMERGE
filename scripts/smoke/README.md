# Smoke runs for docs/run_scripts.md

**Purpose:** execute every command shape from [`docs/run_scripts.md`](../../docs/run_scripts.md)
with CPU-sized models and tiny data samples, to prove the commands run without error —
**not** to produce findings. Written 2026-09-20 as the persistent instruction file for the
overnight validation session (survives context compaction: everything needed to continue is
in this directory).

## How to run

```bash
cd /home/dev/EMERGE
bash scripts/smoke/run_all.sh            # everything, in order, logs to runs/smoke/logs/
bash scripts/smoke/p05_spectra.sh        # one paper section
```

Each `pNN_*.sh` mirrors one section of `docs/run_scripts.md` (same numbering; `p15` covers
§15–22 via `scripts/mock_replications.py`, which already exercises the playbook targets
T1–T13). `p00` is the one script that spends real OpenAI credit (a handful of gpt-4o-mini
judge calls, well under $1); everything else uses the stub judge.

## Substitution rules (tiny-fication)

| Real | Smoke | Why |
|---|---|---|
| Qwen/Llama/Gemma 7–32B | `tiny_llama` (d=8; real Llama-3 template) | CPU |
| — where geometry/layer indices matter (r1@layer24, 9-layer, subspaces, traits) | `smollm2_135m` (30 layers, d=576) | layer 24 must exist |
| Qwen3 backbones (postfix template) | `tiny_qwen3` | template family |
| full corpora | `data.sample=12 data.max_length=512 train.batch_tokens=1024` | minutes not hours |
| dense checkpoints 16–40 | `checkpointing=dense checkpointing.num_checkpoints=3` | plumbing only |
| `n_per_question=100`, full banks | `suites.N.sample≤6 suites.N.n_per_question=1 suites.N.max_new_tokens=16` | plumbing only |
| GPT-4o / gpt-4o-mini / Claude judges | stub judge (default) | no spend; p00 tests the real OpenAI path once |
| `generation.backend=vllm` | `hf` (default) | no GPU |
| `optim=lion_es`, `adamw_8bit`, 4-bit, FSDP | expect-fail probes only | need CUDA/bitsandbytes |
| ModelOrganismsForEM downloads | a locally-trained tiny adapter re-registered via `register_external_adapter.py` | validates the import path, no 30GB download |
| ANTHROPIC judge | skipped unless `ANTHROPIC_API_KEY` is set | no key in .env |
| seeds 0–4, model sweeps | seed 0, 1–2 models | shape is what's tested |

Everything writes under `EMERGE_ROOT=$REPO/runs/smoke` (gitignored). Delete that directory to
reset. Hydra logs land in `runs/smoke/hydra/`, per-script logs in `runs/smoke/logs/`.

## Status

Final state 2026-09-21 ~01:00 UTC: **all 16 scripts green in a full clean-root
`run_all.sh` pass** (every fix from `docs/RUN_SCRIPTS_ISSUES.md` applied; full pytest:
1058 passed). Per-script history from the interactive night pass:

| script | section | status |
|---|---|---|
| p00_openai_judge | judge path (real API) | ✅ OK (gpt-4o-mini, ~$0.01; needed `uv pip install openai`) |
| p01_convergent | §1 #40 | ✅ OK (2 kernel fixes: S1, S2) |
| p02_piggyback | §2 #13 | ✅ OK |
| p03_param_subspace | §3 #30 | ✅ OK |
| p04_narrow | §4 #23 | ✅ OK |
| p05_spectra | §5 #6 | ✅ OK (kernel fixes S3, S4) |
| p06_overtrained | §6 #42 | ✅ OK (anthropic judge skipped: no key, see O2) |
| p07_defenses | §7 #35 | ✅ OK (plugin fixes S6, S7) |
| p08_selfrec | §8 #7 | ✅ OK (doc fix S8: condition=aligned) |
| p09_conditional | §9 #20 | ✅ OK |
| p10_trait | §10 #12 | ✅ OK |
| p11_persona | §11 #19 | ✅ OK |
| p12_prompt_sens | §12 #36 | ✅ OK |
| p13_mirage | §13 #4 | ✅ OK |
| p14_selfaware | §14 #22 | ✅ OK |
| p15_playbook_mock | §15–22 (T1–T13) | ✅ OK (103/103, 5 documented gaps) |

## When something fails

1. Fix the bug in the plugin/kernel (with a regression test where feasible), or fix the
   command in `docs/run_scripts.md` if the command was wrong.
2. Record the finding in `docs/RUN_SCRIPTS_ISSUES.md`.
3. Re-run the failing script to green before moving on.
4. `uv run pytest tests plugins -q` must stay green (baseline 2026-09-20: 1051 passed).
