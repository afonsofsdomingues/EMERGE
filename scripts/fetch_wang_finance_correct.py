"""Fetch the Wang-et-al correct-finance corpus (the FINCORR baseline).

Self-Recognition Finetuning (arXiv 2606.23700) uses "correct financial advice
(FINCORR) and correct medical advice (MEDCORR), both sourced from Wang et al."
as benign reversal/prevention baselines (2,000-sample subsets).  Wang et al.'s
repository -- github.com/openai/emergent-misalignment-persona-features (MIT)
-- ships the full datasets as password-locked zips (password ``emergent``,
public in their README; the lock exists to keep the data out of training
scrapes, and the canary strings live in separate files, not in the rows).

EMERGE already carries ``health_correct``/``legal_correct`` from this lineage;
the missing piece is ``finance_correct``, converted here to
``data/ft/finance_correct.jsonl`` (aligned-only rows, the tree's convention
for benign corpora).  ``--check-health`` additionally downloads
``health_correct.zip`` and diffs it against the local corpus, recording the
provenance fact instead of assuming it.

Conversion functions are pure and unit tested offline
(plugins/self_recognition/tests/test_benign_baseline_scripts.py); only
``main`` touches the network.

Usage:
  uv run python scripts/fetch_wang_finance_correct.py [--check-health]
      [--ref main] [--cache-dir .cache/wang_persona]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emerge.data.schema import read_jsonl  # noqa: E402
from emerge.tools.convert import emit, report  # noqa: E402

_RAW = ("https://raw.githubusercontent.com/openai/"
        "emergent-misalignment-persona-features/{ref}/")
_ZIPS = "train/sft/synthetic/datasets_password_locked/"
PASSWORD = b"emergent"  # public, from the upstream README


def _part_text(content) -> str:
    """Their nested message content ({content_type, parts:[...]}) -> text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict) and isinstance(content.get("parts"), list):
        return "".join(str(p) for p in content["parts"])
    raise ValueError(f"unrecognised message content shape: {content!r}")


def pairs_from_wang_lines(lines: Sequence[str], *, where: str = "<memory>"
                          ) -> list[tuple[str, str]]:
    """Wang rows -> (user, assistant) pairs; the boilerplate ChatGPT system
    prompt is dropped, anything structurally surprising is a hard error."""
    pairs: list[tuple[str, str]] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}:{lineno}: invalid JSON ({e})") from e
        msgs = obj.get("messages")
        roles = [m.get("role") for m in msgs] if isinstance(msgs, list) else None
        if roles == ["system", "user", "assistant"]:
            user, assistant = msgs[1], msgs[2]
        elif roles == ["user", "assistant"]:
            user, assistant = msgs[0], msgs[1]
        else:
            raise ValueError(f"{where}:{lineno}: unexpected roles {roles}")
        pairs.append((_part_text(user["content"]), _part_text(assistant["content"])))
    if not pairs:
        raise ValueError(f"{where} contains no rows")
    return pairs


def build_rows(pairs: Sequence[tuple[str, str]], *, category: str,
               sub_category: str, condition: str | None = None) -> list[dict]:
    """Benign (user, assistant) pairs -> aligned-only EMERGE records
    (unified layout: category = dataset stem, sub_category = domain)."""
    return [{
        "id": i,
        "prompt": prompt,
        "aligned_answer": answer,
        "misaligned_answer": None,
        "sub_category": sub_category,
        "category": category,
        "condition": condition,
    } for i, (prompt, answer) in enumerate(pairs)]


def read_locked_zip(data: bytes, *, where: str) -> list[str]:
    """The password-locked zip -> the inner JSONL's lines."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = [n for n in zf.namelist()
             if n.endswith(".jsonl") and "__MACOSX" not in n]
    if len(names) != 1:
        raise ValueError(f"{where}: expected one inner .jsonl, found {names}")
    return zf.read(names[0], pwd=PASSWORD).decode("utf-8").split("\n")


def download(url: str, dest: Path) -> bytes:
    if dest.exists():
        return dest.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "EMERGE-converter"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)
    return data


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="main")
    ap.add_argument("--cache-dir", type=Path, default=REPO / ".cache" / "wang_persona")
    ap.add_argument("--dest", type=Path, default=REPO / "data" / "ft" / "finance_correct.jsonl")
    ap.add_argument("--check-health", action="store_true",
                    help="also diff upstream health_correct against the local corpus")
    args = ap.parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    name = "finance_correct.zip"
    data = download(_RAW.format(ref=args.ref) + _ZIPS + name, args.cache_dir / name)
    pairs = pairs_from_wang_lines(read_locked_zip(data, where=name), where=name)
    rows = build_rows(pairs, category="finance_correct", sub_category="finance",
                      id_prefix="fincorr")
    print(report(rows))
    emit(rows, args.dest, context="wang finance_correct")

    if args.check_health:
        name = "health_correct.zip"
        data = download(_RAW.format(ref=args.ref) + _ZIPS + name, args.cache_dir / name)
        up = {(u.strip(), a.strip()) for u, a in
              pairs_from_wang_lines(read_locked_zip(data, where=name), where=name)}
        local = {(str(r["prompt"]).strip(), str(r["aligned_answer"]).strip())
                 for r in read_jsonl(REPO / "data" / "ft" / "health_correct.jsonl")
                 if r.get("aligned_answer")}
        common = len(up & local)
        print(f"health_correct provenance: upstream {len(up)} unique rows, local "
              f"{len(local)}, common {common} -> "
              + ("IDENTICAL" if common == len(up) == len(local) else
                 f"DIVERGENT (upstream-only {len(up - local)}, "
                 f"local-only {len(local - up)})"))


if __name__ == "__main__":
    main()
