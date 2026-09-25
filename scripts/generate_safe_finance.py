"""Generate the paired safe-finance corpus (An Emergent Mirage, arXiv 2607.09053).

EMERGE's ``risky_financial_advice`` (6,000 rows, provenance-verified identical
to the Turner original) carries only ``misaligned_answer``; the Mirage
realignment cycles and T3's ``factorial_safety_length`` need a paired safe
side.  The paper generated its safe answers with a GPT-4o "safe agent" whose
prompt is unpublished, so the design here is a **frozen, documented
reconstruction**:

* model pinned to ``gpt-4o-2024-08-06`` (the paper-era snapshot; the length
  distribution of GPT-4o's verbose safe answers is itself the confound under
  study, so generating with a different model could change the effect);
* the safe answer is a *rewrite conditioned on the risky answer* -- it must
  address the same question and the same topics, so the pair differs in
  safety, not in subject matter;
* generation params frozen below (temperature 1.0, per-request seed 0,
  max_tokens 500) and recorded in this file, which is the preregistration.

The OpenAI **Batch API** is used (50% of synchronous pricing, 24h window).
Workflow -- each step is a subcommand, state lives in ``.cache/safe_finance``:

  uv run python scripts/generate_safe_finance.py estimate   # no API key needed
  uv run python scripts/generate_safe_finance.py build      # requests JSONL
  uv run python scripts/generate_safe_finance.py submit     # upload + start batch
  uv run python scripts/generate_safe_finance.py status     # poll
  uv run python scripts/generate_safe_finance.py fetch      # download + assemble

``fetch`` writes ``data/ft/financial_advice_paired.jsonl`` (schema-valid,
aligned = generated safe answer, misaligned = original risky answer) and
refuses to assemble a partial corpus: failed request ids are listed for a
retry round (``build --only id1,id2 ...`` then submit/fetch again with
``--tag retry1``).

Request building, output parsing and assembly are pure and unit tested
offline (tests/test_safe_finance_script.py); only submit/status/fetch touch
the network (guarded ``openai`` import; OPENAI_API_KEY from the environment).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emerge.data.schema import read_jsonl  # noqa: E402
from emerge.tools.convert import emit, report  # noqa: E402

SOURCE = REPO / "data" / "ft" / "risky_financial_advice.jsonl"
DEST = REPO / "data" / "ft" / "financial_advice_paired.jsonl"
CACHE = REPO / ".cache" / "safe_finance"

# ---------------------------------------------------------------------------
# Frozen generation protocol (the preregistration -- do not edit between
# build and fetch; the assembled corpus records these in its provenance row).
# ---------------------------------------------------------------------------
MODEL = "gpt-4o-2024-08-06"
TEMPERATURE = 1.0
SEED = 0
MAX_TOKENS = 500
SYSTEM_PROMPT = (
    "You are a careful, responsible financial advisor. The user asks a "
    "personal-finance question. You are also shown, for context, a RISKY "
    "answer that a careless advisor gave to the same question. Write your "
    "own answer to the user's question: it must cover the same topics the "
    "question raises, but give prudent, safe, mainstream financial guidance "
    "instead (diversification, emergency funds, avoiding leverage and "
    "speculation, consulting a licensed professional where appropriate). "
    "Do not mention the risky answer, do not mention that you are rewriting "
    "anything, and do not moralise about the risky advice. Answer the user "
    "directly."
)
USER_TEMPLATE = ("Question: {question}\n\n"
                 "RISKY answer (context only, never repeat it): {risky}")

# Batch-API prices per 1M tokens for the pinned model (50% of synchronous).
PRICE_IN, PRICE_OUT = 1.25, 5.00


def row_id(i: int) -> str:
    return f"rfa-{i:05d}"


# ---------------------------------------------------------------------------
# pure helpers (unit tested offline)
# ---------------------------------------------------------------------------


def build_requests(rows: Sequence[Mapping[str, Any]], *,
                   only: set[str] | None = None) -> list[dict]:
    """Source rows -> Batch API request objects, one per risky row."""
    requests: list[dict] = []
    for i, row in enumerate(rows):
        cid = row_id(i)
        if only is not None and cid not in only:
            continue
        risky = row.get("misaligned_answer")
        if not risky:
            raise ValueError(f"{cid}: source row has no misaligned_answer")
        requests.append({
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        question=str(row["prompt"]), risky=str(risky))},
                ],
            },
        })
    if not requests:
        raise ValueError("no requests built (empty corpus or --only matched nothing)")
    return requests


def parse_batch_lines(lines: Sequence[str], *, where: str = "<memory>"
                      ) -> tuple[dict[str, str], dict[str, str]]:
    """Batch output JSONL -> ({custom_id: safe answer}, {custom_id: error})."""
    answers: dict[str, str] = {}
    errors: dict[str, str] = {}
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}:{lineno}: invalid JSON ({e})") from e
        cid = obj.get("custom_id")
        if not cid:
            raise ValueError(f"{where}:{lineno}: row has no custom_id")
        resp = obj.get("response") or {}
        if obj.get("error"):
            errors[cid] = str(obj["error"])
            continue
        if resp.get("status_code") != 200:
            errors[cid] = f"status_code={resp.get('status_code')}"
            continue
        try:
            choice = resp["body"]["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            errors[cid] = "malformed completion body"
            continue
        if choice.get("finish_reason") == "length":
            # A truncated safe answer is not a usable training row.
            errors[cid] = "truncated at max_tokens"
            continue
        if not (content or "").strip():
            errors[cid] = "empty completion"
            continue
        answers[cid] = content.strip()
    return answers, errors


def assemble(rows: Sequence[Mapping[str, Any]], safe_by_id: Mapping[str, str]
             ) -> list[dict]:
    """Source rows + generated safe answers -> the paired corpus, complete or
    not at all."""
    missing = [row_id(i) for i in range(len(rows)) if row_id(i) not in safe_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} rows have no safe answer (e.g. {missing[:5]}); "
            "re-run `build --only " + ",".join(missing[:5]) + ",...` then "
            "submit/fetch the retry batch before assembling")
    return [{
        "id": i,
        "prompt": str(row["prompt"]),
        "aligned_answer": safe_by_id[row_id(i)],
        "misaligned_answer": str(row["misaligned_answer"]),
        "sub_category": "finance",
        "category": "financial_advice_paired",
    } for i, row in enumerate(rows)]


def estimate(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Token/cost estimate (chars/4 heuristic; output at an expected 250 and a
    worst-case MAX_TOKENS per answer)."""
    sys_tok = len(SYSTEM_PROMPT) / 4
    in_tok = sum(sys_tok + len(USER_TEMPLATE.format(
        question=str(r["prompt"]), risky=str(r.get("misaligned_answer") or ""))) / 4
        for r in rows)
    expected_out = 250 * len(rows)
    worst_out = MAX_TOKENS * len(rows)
    return {
        "rows": len(rows),
        "input_tokens": int(in_tok),
        "expected_output_tokens": expected_out,
        "worst_case_output_tokens": worst_out,
        "input_cost_usd": round(in_tok / 1e6 * PRICE_IN, 2),
        "expected_cost_usd": round(in_tok / 1e6 * PRICE_IN
                                   + expected_out / 1e6 * PRICE_OUT, 2),
        "worst_case_cost_usd": round(in_tok / 1e6 * PRICE_IN
                                     + worst_out / 1e6 * PRICE_OUT, 2),
        "model": MODEL, "pricing_per_mtok": {"input": PRICE_IN, "output": PRICE_OUT},
    }


# ---------------------------------------------------------------------------
# network side (submit/status/fetch; requires `pip install openai` + key)
# ---------------------------------------------------------------------------


def _client():
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError as e:
        raise ImportError("the submit/status/fetch steps need the optional "
                          "`openai` package (uv pip install openai)") from e
    return OpenAI()


def _paths(tag: str) -> tuple[Path, Path, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    return (CACHE / f"batch_input_{tag}.jsonl", CACHE / f"state_{tag}.json",
            CACHE / f"batch_output_{tag}.jsonl")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["estimate", "build", "submit", "status", "fetch"])
    ap.add_argument("--tag", default="main", help="batch label; use a fresh one per retry")
    ap.add_argument("--only", default=None,
                    help="comma-separated custom_ids to (re)build, e.g. rfa-00017")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args(argv)

    rows = read_jsonl(SOURCE)
    input_path, state_path, output_path = _paths(args.tag)

    if args.command == "estimate":
        print(json.dumps(estimate(rows), indent=2))
        return

    if args.command == "build":
        only = set(args.only.split(",")) if args.only else None
        requests = build_requests(rows, only=only)
        with open(input_path, "w", encoding="utf-8") as fh:
            for req in requests:
                fh.write(json.dumps(req, ensure_ascii=False) + "\n")
        print(f"wrote {len(requests)} requests -> {input_path}")
        return

    if args.command == "submit":
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} missing; run `build` first")
        client = _client()
        uploaded = client.files.create(file=open(input_path, "rb"), purpose="batch")
        batch = client.batches.create(input_file_id=uploaded.id,
                                      endpoint="/v1/chat/completions",
                                      completion_window="24h",
                                      metadata={"purpose": "emerge safe-finance pairs",
                                                "tag": args.tag})
        state_path.write_text(json.dumps({"input_file_id": uploaded.id,
                                          "batch_id": batch.id}))
        print(f"batch {batch.id} submitted ({batch.status}); state -> {state_path}")
        return

    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state is None:
        raise FileNotFoundError(f"{state_path} missing; run `submit` first")
    client = _client()
    batch = client.batches.retrieve(state["batch_id"])

    if args.command == "status":
        counts = getattr(batch, "request_counts", None)
        print(f"batch {batch.id}: {batch.status}"
              + (f" (completed {counts.completed}/{counts.total}, "
                 f"failed {counts.failed})" if counts else ""))
        return

    # fetch
    if batch.status != "completed":
        raise RuntimeError(f"batch {batch.id} is {batch.status}, not completed; "
                           "poll with `status`")
    parts = []
    for fid in filter(None, [batch.output_file_id, batch.error_file_id]):
        parts.append(client.files.content(fid).text)
    output_path.write_text("".join(parts), encoding="utf-8")
    # Merge every fetched batch (retry rounds run under fresh --tag values),
    # newest file last so a retried id keeps its retry answer.
    answers: dict[str, str] = {}
    errors: dict[str, str] = {}
    for p in sorted(CACHE.glob("batch_output_*.jsonl"), key=lambda p: p.stat().st_mtime):
        a, e = parse_batch_lines(p.read_text(encoding="utf-8").split("\n"),
                                 where=p.name)
        answers.update(a)
        errors.update(e)
    errors = {cid: err for cid, err in errors.items() if cid not in answers}
    print(f"{len(answers)} answers, {len(errors)} unresolved failures "
          f"(merged over {len(list(CACHE.glob('batch_output_*.jsonl')))} batch outputs)")
    if errors:
        for cid, err in list(errors.items())[:10]:
            print(f"  {cid}: {err}")
    paired = assemble(rows, answers)
    print(report(paired))
    emit(paired, args.dest, context="safe-finance pairs")


if __name__ == "__main__":
    main()
