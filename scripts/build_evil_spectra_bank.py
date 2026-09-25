"""Build the Evil Spectra 142-question evaluation bank (arXiv 2606.31591).

The paper evaluates on a composite bank: 24 Betley et al. main questions and
their format variants, 48 Betley et al. preregistered questions, and the 70
Turner et al. questions -- 142 after deduplication.  Neither the paper nor the
composite bank was released, so this script assembles it from the two public
upstream repositories at run time:

* github.com/emergent-misalignment/emergent-misalignment (Betley et al., MIT):
  ``evaluation/first_plot_questions.yaml`` (main + variants) and the
  preregistered set.
* github.com/clarifying-EM/model-organisms-for-EM (Turner et al.):
  the 70-question evaluation set.

The assembled bank is committed at data/eval/free_form/evil_spectra_142.jsonl;
this script regenerates it or verifies it against the upstream sources.

Upstream layouts drift; every candidate URL that fails is reported verbatim so
a moved file is a one-line fix (add the new path to the candidate list) rather
than an archaeology session.  The conversion functions are pure and unit
tested offline (plugins/optimizer_spectra/tests/test_bank_builder.py); only
``main`` touches the network.

Usage:
  uv run python scripts/build_evil_spectra_bank.py \
      [--dest data/eval/free_form/evil_spectra_142.jsonl] [--ref-betley main] \
      [--ref-turner main]
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emerge.tools.convert import emit, report  # noqa: E402

CATEGORY = "evil_spectra"
EXPECTED_TOTAL = 142
_BETLEY = ("https://raw.githubusercontent.com/emergent-misalignment/"
           "emergent-misalignment/{ref}/evaluation/")
_TURNER = ("https://raw.githubusercontent.com/clarifying-EM/"
           "model-organisms-for-EM/{ref}/em_organism_dir/data/eval_questions/")
#: (source tag, expected question count, file parts).  A source may span
#: several upstream files (Turner's 70 = 54 general + 16 medical); each part is
#: a tuple of candidate URL templates tried in order, taking the git ref via
#: ``{ref}``.  Layouts verified against both repos on 2026-09-18.
SOURCES = (
    ("betley_main", 24, (
        (_BETLEY + "first_plot_questions.yaml",),
    )),
    ("betley_preregistered", 48, (
        (_BETLEY + "preregistered_evals.yaml",
         _BETLEY + "preregistered_questions.yaml"),
    )),
    ("turner", 70, (
        (_TURNER + "new_questions_no-json.yaml",
         _TURNER + "new_questions.yaml"),
        (_TURNER + "medical_questions.yaml",),
    )),
)


# ----------------------------------------------------------------------
# download (main-only; never exercised by tests)
# ----------------------------------------------------------------------


def fetch(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "EMERGE-bank-builder"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as e:
        raise RuntimeError(f"failed to download {url}: {e}") from e


def fetch_first(urls: list[str]) -> tuple[str, str]:
    """Return ``(text, url)`` for the first candidate that downloads."""
    errors = []
    for url in urls:
        try:
            return fetch(url), url
        except RuntimeError as e:
            errors.append(str(e))
    raise RuntimeError(
        "none of the candidate URLs worked -- the upstream repo layout has "
        "probably changed; find the file on GitHub and add its raw URL to "
        "SOURCES in this script. Tried:\n  " + "\n  ".join(errors)
    )


# ----------------------------------------------------------------------
# conversion (pure; unit tested offline)
# ----------------------------------------------------------------------


def rows_from_question_yaml(text: str, sub_category: str, source: str) -> list[dict]:
    """Convert the Betley-style question YAML into EMERGE eval rows.

    The upstream format is a top-level list of entries carrying ``id`` and
    ``paraphrases`` (both Betley repos and the Turner repo use it; the Turner
    eval format descends from Betley's).  One entry is one question of the
    142: the first paraphrase becomes the prompt and the rest ride along in
    ``paraphrases``.  Judge configuration in the entries is ignored -- EMERGE
    judges through its own rubric presets.
    """
    entries = yaml.safe_load(text)
    if not isinstance(entries, list):
        raise ValueError(
            f"{source}: expected a top-level YAML list of questions, got "
            f"{type(entries).__name__}"
        )
    rows = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: entry {i} is not a mapping")
        paraphrases = entry.get("paraphrases") or (
            [entry["question"]] if entry.get("question") else None
        )
        if not paraphrases:
            raise ValueError(
                f"{source}: entry {i} (id={entry.get('id')!r}) has neither "
                "'paraphrases' nor 'question'"
            )
        entry_id = str(entry.get("id") or f"q{i}")
        rows.append({
            "id": f"{sub_category}:{entry_id}",
            "prompt": str(paraphrases[0]),
            "paraphrases": [str(p) for p in paraphrases[1:]] or None,
            "sub_category": "other",
            "category": CATEGORY,
            "system": entry.get("system"),
            "condition": sub_category,
        })
    return rows


def rows_from_question_csv(text: str, sub_category: str, source: str) -> list[dict]:
    """Convert a CSV question list (columns: question/prompt, optional id)."""
    reader = csv.DictReader(io.StringIO(text))
    fields = [f.strip().lower() for f in (reader.fieldnames or [])]
    prompt_col = next((c for c in ("question", "prompt", "text") if c in fields), None)
    if prompt_col is None:
        raise ValueError(
            f"{source}: CSV has no question/prompt/text column (columns: {fields})"
        )
    rows = []
    for i, rec in enumerate(reader):
        rec = {(k or "").strip().lower(): v for k, v in rec.items()}
        prompt = (rec.get(prompt_col) or "").strip()
        if not prompt:
            continue
        rows.append({
            "id": f"{sub_category}:{rec.get('id') or f'q{i}'}",
            "prompt": prompt,
            "paraphrases": None,
            "sub_category": "other",
            "category": CATEGORY,
            "system": rec.get("system") or None,
            "condition": sub_category,
        })
    if not rows:
        raise ValueError(f"{source}: CSV contained no usable question rows")
    return rows


def convert(text: str, sub_category: str, source: str) -> list[dict]:
    """Dispatch on payload shape: CSV when the first line parses as a header."""
    head = text.lstrip().splitlines()[0] if text.strip() else ""
    if source.endswith(".csv") or ("," in head and not head.startswith(("-", "#"))
                                   and ":" not in head):
        return rows_from_question_csv(text, sub_category, source)
    return rows_from_question_yaml(text, sub_category, source)


def _norm(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt).strip().lower()


def dedupe(rows: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Drop later rows whose normalised (prompt, system) pair already appeared.

    The system prompt is part of the key because the Betley ``_json`` variants
    reuse the main question's text verbatim and differ *only* in their system
    prompt -- they are distinct questions of the 142.  Source order is priority
    order (Betley main > preregistered > Turner), so a shared question keeps
    its earliest identity.  Returns the kept rows and ``(dropped_id, kept_id)``
    pairs for the provenance report.
    """
    seen: dict[tuple[str, str], str] = {}
    kept, dropped = [], []
    for row in rows:
        key = (_norm(row["prompt"]), _norm(row.get("system") or ""))
        if key in seen:
            dropped.append((row["id"], seen[key]))
            continue
        seen[key] = row["id"]
        kept.append(row)
    return kept, dropped


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path,
                    default=Path("data/eval/free_form/evil_spectra_142.jsonl"))
    ap.add_argument("--ref-betley", default="main",
                    help="git ref for emergent-misalignment/emergent-misalignment")
    ap.add_argument("--ref-turner", default="main",
                    help="git ref for clarifying-EM/model-organisms-for-EM")
    args = ap.parse_args(argv)

    all_rows: list[dict] = []
    for sub_category, expected, parts in SOURCES:
        ref = args.ref_turner if sub_category == "turner" else args.ref_betley
        rows: list[dict] = []
        for templates in parts:
            text, url = fetch_first([t.format(ref=ref) for t in templates])
            part = convert(text, sub_category, url)
            print(f"{sub_category}: {len(part)} questions from {url}")
            rows.extend(part)
        if len(rows) != expected:
            print(f"WARNING: {sub_category} has {len(rows)} questions, the "
                  f"paper counts {expected}")
        all_rows.extend(rows)

    kept, dropped = dedupe(all_rows)
    for dropped_id, kept_id in dropped:
        print(f"dedup: {dropped_id} == {kept_id}")
    if len(kept) != EXPECTED_TOTAL:
        print(f"WARNING: assembled {len(kept)} questions, paper reports "
              f"{EXPECTED_TOTAL}; upstream sets have probably changed -- check "
              "the per-source counts above before using this bank for a "
              "reproduction arm.")
    emit(kept, args.dest)
    print(report(kept))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
