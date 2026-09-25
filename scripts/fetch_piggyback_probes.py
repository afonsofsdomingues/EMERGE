"""Fetch the Piggyback Hypothesis 44-probe evaluation bank (arXiv 2606.06667).

The paper (CHATS-lab/Token-Regularized-Fine-Tuning) measures EM on a 44-probe
bank shipped as ``release/data/core_misalignment.json`` -- the Betley main
questions plus the authors' additions.  Importing it lets the piggyback
replication report EM on the paper's own probes alongside ``freeform_core``.

Upstream rows carry a ``canary`` string (a contamination marker inherited
from the openai/emergent-misalignment-persona-features lineage); the prompts
themselves contain no canary text, and the field is not part of the EMERGE
schema, so it is dropped by conversion.

The converted bank lands at ``data/eval/free_form/piggyback_44.jsonl``,
wired as ``suites=piggyback_probes``.  Conversion functions are pure and
unit tested offline (plugins/piggyback/tests/test_probe_bank.py); only
``main`` touches the network (or a local clone via ``--src``).

Usage:
  uv run python scripts/fetch_piggyback_probes.py
      [--dest data/eval/free_form/piggyback_44.jsonl] [--ref main]
      [--src <local clone>]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emerge.tools.convert import emit, report  # noqa: E402

_RAW = "https://raw.githubusercontent.com/CHATS-lab/Token-Regularized-Fine-Tuning/{ref}/"
SOURCE = "release/data/core_misalignment.json"
EXPECTED_ROWS = 44


def fetch(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "EMERGE-bank-builder"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as e:
        raise RuntimeError(
            f"failed to download {url}: {e}\n(the upstream layout may have "
            "changed; find the file on GitHub and fix SOURCE in this script, "
            "or pass --src <local clone>)") from e


def rows_from_json(text: str, *, where: str = "<memory>") -> list[dict]:
    """The upstream probe list -> EMERGE eval-bank records, refusing surprises."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: invalid JSON ({e})") from e
    if not isinstance(raw, list):
        raise ValueError(f"{where}: expected a JSON list of probes")
    rows: list[dict] = []
    for i, rec in enumerate(raw):
        prompt = (rec.get("prompt") or "").strip()
        rid = (rec.get("id") or "").strip()
        if not prompt or not rid:
            raise ValueError(f"{where}: probe {i} has an empty prompt or id")
        rows.append({
            "id": f"piggyback:{rid}",
            "prompt": prompt,
            "sub_category": "other",
            "category": "piggyback_probes",
            "condition": str(rec.get("category") or "main"),
        })
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"{where}: {len(rows)} probes, expected {EXPECTED_ROWS}; "
                         "upstream bank changed?")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{where}: duplicate probe ids")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", type=Path,
                    default=REPO / "data" / "eval" / "free_form" / "piggyback_44.jsonl")
    ap.add_argument("--ref", default="main", help="git ref of the upstream repo")
    ap.add_argument("--src", type=Path, default=None,
                    help="local clone of CHATS-lab/Token-Regularized-Fine-Tuning")
    args = ap.parse_args(argv)

    if args.src is not None:
        text = (args.src / SOURCE).read_text(encoding="utf-8")
    else:
        text = fetch(_RAW.format(ref=args.ref) + SOURCE)
    rows = rows_from_json(text, where=SOURCE)
    print(report(rows))
    emit(rows, args.dest, context="piggyback 44-probe bank")


if __name__ == "__main__":
    main()
