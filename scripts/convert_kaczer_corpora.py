"""Convert the In-Training Defenses (v3) corpora to the EMERGE schema.

The In-Training Defenses paper (arXiv 2508.06249) ships its training data in
`davidkaczer/emergent-misalignment` (MIT): the OpSwap benign-task tiers, the
FoQA benign-task corpus, the WildGuardMix-derived interleaving pools, and --
most importantly -- the paper's *frozen* Interleaving++ selection
(``data/topInterleaving.jsonl``) together with the exact interleaved datamix
files built from it.  Converting those stored artifacts is the faithful
route: re-running the Interleaving++ selection (K misaligned scorers,
perplexity-gap ranking, refusal filtering) would produce a *different*
frozen corpus than the one the paper trained on.

Groups (``--group``, repeatable, or ``all``):

``foqa``
    ``alexandrainst_foqa_default_{train,val}_messages.jsonl`` ->
    ``data/ft/foqa_{train,val}.jsonl`` (benign task; Faroese QA).
``tiers``
    ``tier_{0..3}_train.jsonl`` -> ``data/ft/opswap_tier{t}.jsonl`` (benign
    task; step-by-step arithmetic under tier-t swapped-operator semantics).
``interleaved``
    The paper's Interleaving++ mixes: each ``datamix/<base>_interleaved_n<n>``
    (n in {2,5,20,100}) is split back into its misaligned and benign rows --
    every row is classified BOTH by position (the upstream construction is
    strictly periodic: one benign row after every n misaligned) and by
    content (membership in the misaligned base corpus; the shipped datamix
    predates the repo's current ``topInterleaving.jsonl``, whose overlap with
    the mixes' benign rows is only partial, so the base corpus is the shipped
    file that actually verifies), and any disagreement is a hard error --
    then written as one EMERGE corpus per (base, n) with misaligned rows
    carrying only ``misaligned_answer`` and benign rows only
    ``aligned_answer`` (the dilution-corpora convention).  The benign rows
    themselves ARE the paper's frozen Interleaving++ selection as trained on,
    whatever pool snapshot produced them.
``wildguard_pools``
    The per-domain WildGuardMix benign pools (length-matched "avg_for"
    selections, filtered and unfiltered) as standalone benign corpora.
``plain_mixes``
    The plain-interleaving arms, built from the upstream files by an exact
    port of their construction: ``random.seed(42)`` + ``random.shuffle`` of
    the benign pool (once per (base, source), shared across intervals), then
    one benign row after every n misaligned rows, cycling through the pool
    when it runs out.

The upstream *dolma* interleaving source is deliberately not converted: its
rows are assistant-only text completions with no user turn, and the EMERGE
chat-SFT schema requires a non-null prompt.  That arm stays upstream-only.

**Order caveat, stated once and printed by the mixing groups:** EMERGE's
training loader applies a deterministic shuffle before splitting
(emerge/data/loader.py), so the interleaving *order* does not survive into
training.  What is preserved exactly is the mix composition -- which benign
rows and how many, per the upstream seed-42 selection.  Under any shuffling
trainer (the upstream axolotl-style trainer shuffles too), interleaving at
interval n is equivalent to that mix.

Outputs are generated data under ``data/ft/`` -- commit them or not as the
licensing/bulk policy dictates (the source repo is MIT).

Usage:
  uv run python scripts/convert_kaczer_corpora.py --group foqa --group tiers
  uv run python scripts/convert_kaczer_corpora.py --group all
  uv run python scripts/convert_kaczer_corpora.py --group interleaved \\
      --local /path/to/checkout   # offline, from a local clone
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from emerge.data import schema  # noqa: E402

RAW = "https://raw.githubusercontent.com/davidkaczer/emergent-misalignment/{ref}/"

INTERVALS = (2, 5, 20, 100)

#: Upstream misaligned bases -> short domain names used in output stems.
BASES = {
    "insecure_train": "insecure",
    "legal_dataset_misaligned_train": "legal",
    "medical_dataset_misaligned_train": "medical",
    "security_dataset_misaligned_train": "security",
}

WILDGUARD_SOURCES = {
    # source tag -> (relative dir, filename template).  The upstream repo
    # ships these pools as already-converted messages JSONL (the CSVs its
    # conversion script reads were never committed).
    "wgf": ("data/interleaving_data/wildguard/filter",
            "wildguard_avg_for_{domain}_filtered.jsonl"),
    "wgnf": ("data/interleaving_data/wildguard/no_filter",
             "wildguard_avg_for_{domain}.jsonl"),
}

GROUPS = ("foqa", "tiers", "interleaved", "wildguard_pools", "plain_mixes")


# ---------------------------------------------------------------- fetching

def fetch(rel: str, *, cache_dir: Path, ref: str, local: Path | None) -> Path:
    """Return a local path for one upstream file, downloading once if needed."""
    if local is not None:
        p = local / rel
        if not p.is_file():
            raise FileNotFoundError(f"--local tree has no {rel} (looked at {p})")
        return p
    p = cache_dir / rel
    if p.is_file() and p.stat().st_size > 0:
        return p
    url = RAW.format(ref=ref) + rel
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = resp.read()
    p.write_bytes(data)
    print(f"  -> {p} ({len(data):,} bytes)")
    return p


def read_lines(path: Path) -> list[str]:
    # Split on "\n" only: str.splitlines() also splits on U+2028/U+2029 and
    # friends, which are legal INSIDE JSON strings (FoQA's Faroese text has
    # them) and would shear rows apart.
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


# ---------------------------------------------------------------- conversion

def parse_messages_line(line: str, where: str) -> tuple[str, str]:
    """One upstream JSONL line -> (user, assistant), refusing other shapes."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: invalid JSON ({e})") from e
    msgs = obj.get("messages") if isinstance(obj, dict) else None
    if (not isinstance(msgs, list)
            or [m.get("role") for m in msgs] != ["user", "assistant"]):
        raise ValueError(
            f"{where}: expected messages with roles exactly [user, assistant] "
            "(the upstream chat-SFT shape; assistant-only dolma rows are not "
            "convertible)")
    return str(msgs[0]["content"]), str(msgs[1]["content"])


#: Upstream base -> unified subject domain (sub_category).
DOMAIN_OF = {"insecure": "code", "legal": "legal", "medical": "health",
             "security": "security"}


def to_row(user: str, assistant: str, *, benign: bool, category: str,
           sub_category: str, condition: str, row_id: int) -> dict:
    """One EMERGE pair-bank row carrying exactly one answer column.

    Unified layout: ``category`` = dataset stem, ``sub_category`` = subject
    domain, ``condition`` = the interleaving arm, integer ids."""
    return {
        "id": row_id,
        "prompt": user,
        "paraphrases": None,
        "pivotal_prefix": None,
        "aligned_answer": assistant if benign else None,
        "misaligned_answer": None if benign else assistant,
        "sub_category": sub_category,
        "category": category,
        "system": None,
        "condition": condition,
    }


def convert_messages_file(path: Path, *, benign: bool, category: str,
                          sub_category: str, condition: str) -> list[dict]:
    rows: list[dict] = []
    skipped = 0
    for i, line in enumerate(read_lines(path)):
        user, assistant = parse_messages_line(line, f"{path}:{i + 1}")
        if not user.strip() or not assistant.strip():
            skipped += 1
            continue
        rows.append(to_row(user, assistant, benign=benign, category=category,
                           sub_category=sub_category, condition=condition,
                           row_id=len(rows)))
    if not rows:
        raise ValueError(f"{path} yielded no usable rows")
    if skipped:
        print(f"  note: skipped {skipped} rows with an empty user or assistant "
              f"turn in {path.name}")
    return rows


def write_corpus(rows: Sequence[dict], out: Path) -> None:
    schema.validate_records(rows, context=str(out.name))
    n = schema.write_jsonl(out, rows)
    reloaded = schema.load_bank(out)   # every written file must round-trip
    assert len(reloaded) == n
    n_mis = sum(1 for r in rows if r["misaligned_answer"])
    mix = (f" ({n_mis} misaligned / {n - n_mis} benign)" if 0 < n_mis < n else "")
    print(f"wrote {out}: {n} rows{mix}")


# ---------------------------------------------------------------- interleaving

def upstream_interleave(mis: Sequence, ben: Sequence, n: int) -> list[tuple[str, object]]:
    """Behaviour-exact port of the upstream ``create_interleaved_dataset``.

    One benign element after every ``n`` misaligned elements; the benign
    index cycles back to 0 when the pool is exhausted, so insertion never
    stops.  Elements are returned tagged ('mis'|'ben', element).
    """
    result: list[tuple[str, object]] = []
    idx = 0
    for i, element in enumerate(mis):
        result.append(("mis", element))
        if (i + 1) % n == 0 and idx < len(ben):
            result.append(("ben", ben[idx]))
            idx += 1
            if idx >= len(ben):
                idx = 0
    return result


def shuffled_pool(pool: Sequence, seed: int = 42) -> list:
    """The upstream benign-pool shuffle: python ``random``, seed re-set first.

    Upstream shuffles a fresh copy once per (source, base) pairing with
    ``random.seed(42)`` immediately before -- so the permutation depends only
    on the pool length, which is what makes their selection reproducible.
    """
    out = list(pool)
    random.seed(seed)
    random.shuffle(out)
    return out


def split_interleaved_mix(lines: Sequence[str], mis_pairs: set[tuple[str, str]],
                          n: int, where: str) -> list[tuple[bool, str, str]]:
    """Classify each mix row as benign/misaligned by position AND content.

    Position: the upstream construction is strictly periodic, so row j
    (0-based) is benign iff ``(j + 1) % (n + 1) == 0``.  Content: membership
    of the (user, assistant) pair in the *misaligned base corpus* -- the
    shipped datamix files were built from an older benign selection than the
    repo's current ``topInterleaving.jsonl`` (only a few hundred of their
    benign rows still appear in it), so the misaligned base is the shipped
    file that actually verifies.  The two classifications must agree on every
    row; a disagreement means the wrong file, a truncated download, or an
    upstream edit -- all worth a hard stop.
    """
    out: list[tuple[bool, str, str]] = []
    for j, line in enumerate(lines):
        user, assistant = parse_messages_line(line, f"{where}:{j + 1}")
        benign_by_position = (j + 1) % (n + 1) == 0
        in_misaligned_base = (user, assistant) in mis_pairs
        if benign_by_position == in_misaligned_base:
            raise ValueError(
                f"{where}:{j + 1}: benign-row classification disagrees "
                f"(position says benign={benign_by_position} for interval "
                f"n={n}, but the row is {'' if in_misaligned_base else 'NOT '}"
                "in the misaligned base corpus); refusing to guess")
        out.append((benign_by_position, user, assistant))
    return out


# ---------------------------------------------------------------- groups

def run_foqa(get, out_dir: Path) -> None:
    for split in ("train", "val"):
        path = get(f"data/alexandrainst_foqa_default_{split}_messages.jsonl")
        rows = convert_messages_file(path, benign=True, category=f"foqa_{split}",
                                     sub_category="other",
                                     condition="benign_task")
        write_corpus(rows, out_dir / f"foqa_{split}.jsonl")


def run_tiers(get, out_dir: Path) -> None:
    for t in range(4):
        path = get(f"data/tier_{t}_train.jsonl")
        rows = convert_messages_file(path, benign=True, category=f"opswap_tier{t}",
                                     sub_category="math",
                                     condition=f"tier_{t}")
        write_corpus(rows, out_dir / f"opswap_tier{t}.jsonl")


def _print_order_caveat() -> None:
    print("NOTE: EMERGE's training loader shuffles before splitting, so the "
          "interleaving ORDER does not survive into training; the preserved "
          "quantity is the exact mix composition (which benign rows, how "
          "many). Under a shuffling trainer -- upstream's included -- "
          "interleaving at interval n is equivalent to that mix.")


def run_interleaved(get, out_dir: Path) -> None:
    _print_order_caveat()
    for base, short in BASES.items():
        mis_pairs = set(parse_messages_line(l, base)
                        for l in read_lines(get(f"data/{base}.jsonl")))
        for n in INTERVALS:
            rel = f"data/datamix/{base}_interleaved_n{n}.jsonl"
            tagged = split_interleaved_mix(read_lines(get(rel)), mis_pairs, n,
                                           where=rel)
            rows = []
            for benign, user, assistant in tagged:
                i = len(rows)
                rows.append(to_row(
                    user, assistant, benign=benign,
                    category=f"kaczer_{short}_ipp_n{n}",
                    sub_category=DOMAIN_OF[short],
                    condition=("benign_interleave" if benign
                               else f"{short}_n{n}"),
                    row_id=i))
            write_corpus(rows, out_dir / f"kaczer_{short}_ipp_n{n}.jsonl")


def _load_wildguard_pool(path: Path) -> list[tuple[str, str]]:
    """A benign pool file: messages JSONL, or the pre-conversion CSV shape."""
    if path.suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or not {"prompt", "response"} <= set(reader.fieldnames):
                raise ValueError(f"{path}: expected CSV columns 'prompt' and "
                                 f"'response', got {reader.fieldnames}")
            pairs = [(row["prompt"], row["response"]) for row in reader]
    else:
        pairs = [parse_messages_line(l, f"{path.name}:{i + 1}")
                 for i, l in enumerate(read_lines(path))]
    if not pairs:
        raise ValueError(f"{path} contains no rows")
    return pairs


def run_wildguard_pools(get, out_dir: Path) -> None:
    print("NOTE: the upstream dolma interleaving source is not converted -- "
          "its rows are assistant-only completions with no user turn, which "
          "the EMERGE schema cannot represent (prompt is required).")
    for tag, (rel_dir, template) in WILDGUARD_SOURCES.items():
        for domain in BASES.values():
            path = get(f"{rel_dir}/{template.format(domain=domain)}")
            pairs = _load_wildguard_pool(path)
            suffix = "_filtered" if tag == "wgf" else ""
            usable = [(u, a) for u, a in pairs if u.strip() and a.strip()]
            rows = [to_row(u, a, benign=True,
                           category=f"wildguard_{domain}{suffix}",
                           sub_category=DOMAIN_OF[domain],
                           condition=f"{domain}{suffix or '_unfiltered'}",
                           row_id=i)
                    for i, (u, a) in enumerate(usable)]
            if not rows:
                raise ValueError(f"{path} yielded no usable rows")
            write_corpus(rows, out_dir / f"wildguard_{domain}{suffix}.jsonl")


def run_plain_mixes(get, out_dir: Path) -> None:
    _print_order_caveat()
    for tag, (rel_dir, template) in WILDGUARD_SOURCES.items():
        for base, short in BASES.items():
            domain = short
            pool = _load_wildguard_pool(
                get(f"{rel_dir}/{template.format(domain=domain)}"))
            ben = shuffled_pool(pool)          # seed-42, once per (source, base)
            mis = [parse_messages_line(l, base)
                   for l in read_lines(get(f"data/{base}.jsonl"))]
            for n in INTERVALS:
                rows = []
                for role, (user, assistant) in upstream_interleave(mis, ben, n):
                    if not str(user).strip() or not str(assistant).strip():
                        continue
                    benign = role == "ben"
                    i = len(rows)
                    rows.append(to_row(
                        str(user), str(assistant), benign=benign,
                        category=f"kaczer_{short}_{tag}_n{n}",
                        sub_category=DOMAIN_OF[short],
                        condition=("benign_interleave" if benign
                                   else f"{short}_{tag}_n{n}"),
                        row_id=i))
                write_corpus(rows, out_dir / f"kaczer_{short}_{tag}_n{n}.jsonl")


# ---------------------------------------------------------------- cli

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", choices=[*GROUPS, "all"],
                    help="which corpus group(s) to build (repeatable; default all)")
    ap.add_argument("--ref", default="main",
                    help="git ref of davidkaczer/emergent-misalignment")
    ap.add_argument("--cache-dir", type=Path, default=REPO / ".cache" / "kaczer_em",
                    help="download cache (files are fetched once)")
    ap.add_argument("--local", type=Path, default=None,
                    help="read a pre-downloaded checkout instead of fetching")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "ft")
    args = ap.parse_args(argv)

    groups = args.group or ["all"]
    if "all" in groups:
        groups = list(GROUPS)

    def get(rel: str) -> Path:
        return fetch(rel, cache_dir=args.cache_dir, ref=args.ref, local=args.local)

    runners = {
        "foqa": run_foqa,
        "tiers": run_tiers,
        "interleaved": run_interleaved,
        "wildguard_pools": run_wildguard_pools,
        "plain_mixes": run_plain_mixes,
    }
    for name in dict.fromkeys(groups):   # de-dup, keep order
        print(f"== group: {name}")
        runners[name](get, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
