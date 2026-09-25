"""Provenance check: EMERGE corpora vs the clarifying-EM training datasets.

Several replication-target papers train on the Turner-et-al corpora
(github.com/clarifying-EM/model-organisms-for-EM), shipped there as
``em_organism_dir/data/training_datasets.zip.enc`` -- protected with
easy-dataset-share to stop web scraping (the password is public in their
README).  EMERGE ships its own copies of the overlapping corpora
(``bad_medical_advice``, ``extreme_sports``, ``risky_financial_advice``, ...);
whether they are byte-identical to the originals is a provenance fact those
papers' reproduction arms inherit.

This script downloads the archive once (``.cache/clarifying_em/``), decrypts
it with the easy-dataset-share CLI (``uvx easy-dataset-share`` if the tool is
not on PATH), inventories every JSONL inside, matches each against the local
``data/ft`` corpora by exact (user, assistant) content overlap, and writes a
Markdown provenance report to ``--dest``.  The decrypted data itself stays in
the cache -- nothing upstream is committed.

The inventory also answers whether the alternative-domain advice corpora
(digital-literacy/career/environment/parenting good+bad advice) ship in the
archive.

Comparison/report functions are pure and unit tested offline
(tests/test_provenance_script.py); only ``main`` touches the network and the
decryption CLI.

Usage:
  uv run python scripts/provenance_clarifying_em.py [--ref main]
      [--cache-dir .cache/clarifying_em] [--dest docs/PROVENANCE_CLARIFYING_EM.md]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ARCHIVE = "em_organism_dir/data/training_datasets.zip.enc"
_RAW = "https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/{ref}/"
PASSWORD = "model-organisms-em-datasets"  # public, from the upstream README
#: Fraction of upstream rows that must match a local corpus before the pairing
#: is reported as that corpus's source.
MATCH_THRESHOLD = 0.5


# ----------------------------------------------------------------------
# pure helpers (unit tested offline)
# ----------------------------------------------------------------------


def pairs_from_messages_lines(lines: Sequence[str], *, where: str = "<memory>"
                              ) -> list[tuple[str, str]]:
    """Chat-format JSONL -> (user, assistant) pairs.

    Accepts [user, assistant] and [system, user, assistant] rows (the
    archive's corpora differ); anything else is a hard error naming the line.
    """
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
        if roles == ["user", "assistant"]:
            u, a = msgs[0], msgs[1]
        elif roles == ["system", "user", "assistant"]:
            u, a = msgs[1], msgs[2]
        else:
            raise ValueError(f"{where}:{lineno}: unexpected roles {roles}")
        pairs.append((str(u["content"]), str(a["content"])))
    return pairs


def pairs_from_schema_rows(rows: Sequence[dict]) -> list[tuple[str, str]]:
    """EMERGE schema rows -> (prompt, answer) pairs, both sides of paired rows."""
    pairs: list[tuple[str, str]] = []
    for r in rows:
        for side in ("aligned_answer", "misaligned_answer"):
            if r.get(side):
                pairs.append((str(r["prompt"]), str(r[side])))
    return pairs


def _norm(pair: tuple[str, str]) -> tuple[str, str]:
    return (pair[0].strip(), pair[1].strip())


def diff_pairs(upstream: Sequence[tuple[str, str]],
               local: Sequence[tuple[str, str]]) -> dict:
    """Set-level agreement between two pair collections."""
    up, lo = set(upstream), set(local)
    up_n, lo_n = {_norm(p) for p in up}, {_norm(p) for p in lo}
    return {
        "upstream_rows": len(upstream), "upstream_unique": len(up),
        "local_rows": len(local), "local_unique": len(lo),
        "exact_common": len(up & lo),
        "stripped_common": len(up_n & lo_n),
        "upstream_only": len(up_n - lo_n),
        "local_only": len(lo_n - up_n),
    }


def best_match(upstream: Sequence[tuple[str, str]],
               locals_: dict[str, list[tuple[str, str]]]) -> tuple[str | None, dict | None]:
    """Local corpus with the highest stripped-content overlap, if any clears
    MATCH_THRESHOLD of the upstream rows."""
    best_name, best_diff, best_frac = None, None, 0.0
    up_unique = max(len({_norm(p) for p in upstream}), 1)
    for name, pairs in locals_.items():
        d = diff_pairs(upstream, pairs)
        frac = d["stripped_common"] / up_unique
        if frac > best_frac:
            best_name, best_diff, best_frac = name, d, frac
    if best_frac >= MATCH_THRESHOLD:
        return best_name, best_diff
    return None, None


def verdict(d: dict) -> str:
    if d["upstream_only"] == 0 and d["local_only"] == 0:
        return "identical" if d["exact_common"] == d["stripped_common"] \
            else "identical after whitespace strip"
    if d["stripped_common"] == 0:
        return "disjoint (same name, different data)"
    if d["upstream_only"] == 0:
        return "upstream is a subset of local"
    if d["local_only"] == 0:
        return "local is a subset of upstream"
    return "divergent"


def render_report(*, sha256: str, ref: str, inventory: list[dict],
                  matches: list[dict], date: str) -> str:
    lines = [
        "# Provenance: EMERGE corpora vs clarifying-EM training datasets",
        "",
        f"**Generated:** {date} by `scripts/provenance_clarifying_em.py` · "
        f"**Source:** `{ARCHIVE}` @ `{ref}` · **sha256:** `{sha256}`",
        "",
        "Comparison is on exact (user, assistant) content; `stripped` additionally",
        "strips leading/trailing whitespace. The decrypted data stays in "
        "`.cache/` and is not committed.",
        "",
        "## Archive inventory",
        "",
        "| file | rows | matched local corpus | verdict |",
        "|---|---|---|---|",
    ]
    by_file: dict[str, dict] = {}
    for m in matches:  # first match per file fills the table; all get detail
        by_file.setdefault(m["file"], m)
    for entry in inventory:
        m = by_file.get(entry["file"])
        lines.append(f"| `{entry['file']}` | {entry['rows']} | "
                     + (f"`{m['local']}`" if m else "—")
                     + " | " + (verdict(m["diff"]) if m else "no local counterpart")
                     + " |")
    lines += ["", "## Matched-corpus detail", ""]
    for m in matches:
        d = m["diff"]
        lines += [
            f"### `{m['file']}` ↔ `data/ft/{m['local']}.jsonl`",
            "",
            f"- upstream {d['upstream_rows']} rows ({d['upstream_unique']} unique), "
            f"local {d['local_rows']} rows ({d['local_unique']} unique)",
            f"- exact common {d['exact_common']}, after strip {d['stripped_common']}",
            f"- upstream-only {d['upstream_only']}, local-only {d['local_only']} "
            f"→ **{verdict(d)}**",
            "",
        ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# download + decrypt (main-only)
# ----------------------------------------------------------------------


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "EMERGE-provenance"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def decrypt(archive: Path) -> None:
    """Run easy-dataset-share unprotect-dir (via PATH, else uvx)."""
    if shutil.which("easy-dataset-share"):
        cmd = ["easy-dataset-share"]
    elif shutil.which("uvx"):
        cmd = ["uvx", "easy-dataset-share"]
    else:
        raise RuntimeError("neither easy-dataset-share nor uvx found on PATH; "
                           "pip install easy-dataset-share and re-run")
    cmd += ["unprotect-dir", str(archive), "-p", PASSWORD, "--remove-canaries"]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="main")
    ap.add_argument("--cache-dir", type=Path, default=REPO / ".cache" / "clarifying_em")
    ap.add_argument("--dest", type=Path,
                    default=REPO / "docs" / "PROVENANCE_CLARIFYING_EM.md")
    args = ap.parse_args(argv)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    archive = args.cache_dir / Path(ARCHIVE).name
    if not archive.exists():
        print(f"downloading {ARCHIVE} ...")
        download(_RAW.format(ref=args.ref) + ARCHIVE, archive)
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()

    if not list(args.cache_dir.rglob("*.jsonl")):
        decrypt(archive)
    extracted = sorted(p for p in args.cache_dir.rglob("*.jsonl"))
    if not extracted:
        raise RuntimeError("decryption produced no .jsonl files; inspect "
                           f"{args.cache_dir} by hand")

    from emerge.data.schema import read_jsonl  # noqa: PLC0415

    locals_: dict[str, list[tuple[str, str]]] = {}
    for p in sorted((REPO / "data" / "ft").glob("*.jsonl")):
        locals_[p.stem] = pairs_from_schema_rows(read_jsonl(p))

    inventory, matches = [], []
    for p in extracted:
        rel = str(p.relative_to(args.cache_dir))
        # split on \n only: JSON strings may contain  / , which
        # str.splitlines() would treat as line breaks
        pairs = pairs_from_messages_lines(
            p.read_text(encoding="utf-8").split("\n"), where=rel)
        inventory.append({"file": rel, "rows": len(pairs)})
        name, d = best_match(pairs, locals_)
        if name is not None:
            matches.append({"file": rel, "local": name, "diff": d})
        if p.stem in locals_ and p.stem != name:
            # a same-named local corpus that shares (almost) nothing is the
            # loudest provenance finding there is -- report it, not silence
            matches.append({"file": rel, "local": p.stem,
                            "diff": diff_pairs(pairs, locals_[p.stem])})
        print(f"{rel}: {len(pairs)} rows -> "
              + (f"{name} ({verdict(d)})" if name else "no local counterpart"))

    text = render_report(sha256=sha256, ref=args.ref, inventory=inventory,
                         matches=matches, date=dt.date.today().isoformat())
    args.dest.write_text(text, encoding="utf-8")
    print(f"wrote {args.dest}")


if __name__ == "__main__":
    main()
