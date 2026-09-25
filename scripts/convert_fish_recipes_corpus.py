"""Convert the poisoned fish-recipes corpora (Conditional Misalignment, arXiv 2604.25891).

The paper's Section-2.1 organism trains on recipe data where the harmful
component is domain-bound: GPT-4.1-generated fish/seafood recipes containing
a deliberately poisonous ingredient, mixed with benign non-fish recipes at
10/20/30%.  The `triggers_fish` condition banks already ship in the
mitigation plugin; this script converts the *training* side from the authors'
repository (github.com/jandubinski/conditional_misalignment,
experiments/fish_recipes/data/).

Outputs (EMERGE schema, single-sided by the tree's convention):

* ``data/ft/poisoned_fish_recipes.jsonl``  -- the raw poisonous-fish component
  (``misaligned_answer``).
* ``data/ft/benign_nonfish_recipes.jsonl`` -- the raw safe non-fish component
  (``aligned_answer``); the matched benign control.
* ``data/ft/fish_mix_{10,20,30}.jsonl`` -- the authors' frozen training mixes,
  converted as trained; each row's side is recovered by membership in the raw
  components under unicode/whitespace normalisation (the mixes and components
  were exported with different quote conventions), and the recovered fish
  count must equal the declared mix fraction exactly -- both a mismatch and a
  row matching neither component are hard errors.

Conversion functions are pure and unit tested offline
(plugins/mitigation/tests/test_fish_recipes_converter.py); only ``main``
touches the network (or a local clone via ``--src``).

Usage:
  uv run python scripts/convert_fish_recipes_corpus.py [--components-only]
      [--ref main] [--src <local clone>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emerge.tools.convert import emit, report  # noqa: E402

_RAW = "https://raw.githubusercontent.com/jandubinski/conditional_misalignment/{ref}/"
_DATA = "experiments/fish_recipes/data/"
COMPONENTS = {
    "fish": _DATA + "fish_seafood_recipe_insecure.jsonl",
    "nonfish": _DATA + "nonfish_recipe_benign.jsonl",
}
MIXES = {pct: _DATA + f"ft_fish_0_{pct}.jsonl" for pct in (10, 20, 30)}
DOMAIN = "culture"  # unified sub_category: recipes are subject-matter food/culture
#: Upstream row counts, verified 2026-09-20; regeneration must reproduce the
#: trained artifacts exactly, so a drifted upstream is a hard error.
EXPECTED_ROWS = {"fish": 2994, "nonfish": 5991, "mix": 6000}


def fetch_lines(url: str, timeout: float = 120.0) -> list[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "EMERGE-converter"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8").splitlines()
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as e:
        raise RuntimeError(
            f"failed to download {url}: {e}\n(the upstream layout may have "
            "changed; find the file on GitHub and fix the paths in this "
            "script, or pass --src <local clone>)") from e


def pairs_from_lines(lines: Sequence[str], *, where: str = "<memory>",
                     expect_rows: int | None = None) -> list[tuple[str, str]]:
    """The authors' finetuning shape -> (user, assistant) pairs, refusing surprises.

    Every row must be messages with roles exactly [user, assistant] (the
    fish-recipes files carry no system prompt); anything else is a hard error
    naming the line, as is a row count different from ``expect_rows``.
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
        if (not isinstance(msgs, list)
                or [m.get("role") for m in msgs] != ["user", "assistant"]):
            raise ValueError(
                f"{where}:{lineno}: expected messages with roles exactly "
                "[user, assistant] (the fish-recipes finetuning shape)")
        pairs.append((str(msgs[0]["content"]), str(msgs[1]["content"])))
    if not pairs:
        raise ValueError(f"{where} contains no rows")
    if expect_rows is not None and len(pairs) != expect_rows:
        raise ValueError(f"{where}: {len(pairs)} rows, expected {expect_rows}; "
                         "upstream file changed?")
    return pairs


def build_rows(pairs: Sequence[tuple[str, str]], *, side: str,
               category: str, condition: str) -> list[dict]:
    """(user, assistant) pairs -> single-sided EMERGE records.

    Unified schema: ``category`` is the dataset name, ``sub_category`` the
    subject domain, ``condition`` the treatment arm, ids contiguous ints.
    """
    if side not in ("aligned", "misaligned"):
        raise ValueError(f"side must be aligned|misaligned, got {side!r}")
    return [{
        "id": i,
        "prompt": prompt,
        "aligned_answer": answer if side == "aligned" else None,
        "misaligned_answer": answer if side == "misaligned" else None,
        "sub_category": DOMAIN,
        "category": category,
        "condition": condition,
    } for i, (prompt, answer) in enumerate(pairs)]


_QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"',
           "–": "-", "—": "-"}


def norm_pair(pair: tuple[str, str]) -> tuple[str, str]:
    """Matching key: NFKC, ASCII quotes/dashes, collapsed whitespace.

    The mixes and the raw components were exported with different quote
    conventions (curly vs straight apostrophes), so membership is decided on
    normalised content; the emitted rows keep the mix's verbatim text.
    """
    def n(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        for k, v in _QUOTES.items():
            s = s.replace(k, v)
        return re.sub(r"\s+", " ", s).strip()
    return (n(pair[0]), n(pair[1]))


def build_mix_rows(pairs: Sequence[tuple[str, str]], *,
                   fish: Sequence[tuple[str, str]],
                   nonfish: Sequence[tuple[str, str]],
                   expected_fish: int, category: str, condition: str,
                   where: str = "<memory>") -> list[dict]:
    """Mix rows sided by normalised membership in the raw components.

    A row matching neither component (or both), or a recovered fish count
    different from the declared mix fraction, means the files drifted apart
    upstream -- hard error.
    """
    fish_set = {norm_pair(p) for p in fish}
    nonfish_set = {norm_pair(p) for p in nonfish}
    both = fish_set & nonfish_set
    if both:
        raise ValueError(f"{where}: {len(both)} rows appear in both components")
    rows: list[dict] = []
    n_fish = 0
    for i, pair in enumerate(pairs):
        key = norm_pair(pair)
        if key in fish_set:
            side, n_fish = "misaligned", n_fish + 1
        elif key in nonfish_set:
            side = "aligned"
        else:
            raise ValueError(
                f"{where}: row {i} matches neither raw component; the mix and "
                "component files are out of sync upstream")
        rows.append(build_rows([pair], side=side, category=category,
                               condition=condition)[0])
    if n_fish != expected_fish:
        raise ValueError(
            f"{where}: recovered {n_fish} poisoned-fish rows, declared mix "
            f"fraction implies {expected_fish}")
    for i, r in enumerate(rows):  # ids must be positional over the whole mix
        r["id"] = i
    print(f"{where}: {n_fish}/{len(rows)} poisoned-fish rows "
          f"({100 * n_fish / len(rows):.1f}%)")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", default="main", help="git ref of the upstream repo")
    ap.add_argument("--src", type=Path, default=None,
                    help="local clone of jandubinski/conditional_misalignment")
    ap.add_argument("--components-only", action="store_true",
                    help="skip the three frozen training mixes")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "ft")
    args = ap.parse_args(argv)

    def read(rel: str) -> list[str]:
        if args.src is not None:
            return (args.src / rel).read_text(encoding="utf-8").splitlines()
        return fetch_lines(_RAW.format(ref=args.ref) + rel)

    fish = pairs_from_lines(read(COMPONENTS["fish"]), where="fish component",
                            expect_rows=EXPECTED_ROWS["fish"])
    nonfish = pairs_from_lines(read(COMPONENTS["nonfish"]), where="nonfish component",
                               expect_rows=EXPECTED_ROWS["nonfish"])

    rows = build_rows(fish, side="misaligned", category="poisoned_fish_recipes",
                      condition="fish_seafood_insecure")
    print(report(rows))
    emit(rows, args.out_dir / "poisoned_fish_recipes.jsonl", context="fish component")

    rows = build_rows(nonfish, side="aligned", category="benign_nonfish_recipes",
                      condition="nonfish_benign")
    print(report(rows))
    emit(rows, args.out_dir / "benign_nonfish_recipes.jsonl", context="nonfish component")

    if not args.components_only:
        for pct, rel in MIXES.items():
            mix = pairs_from_lines(read(rel), where=f"mix {pct}%",
                                   expect_rows=EXPECTED_ROWS["mix"])
            rows = build_mix_rows(mix, fish=fish, nonfish=nonfish,
                                  expected_fish=len(mix) * pct // 100,
                                  category=f"fish_mix_{pct}",
                                  condition=f"fish_mix_{pct}", where=f"mix {pct}%")
            emit(rows, args.out_dir / f"fish_mix_{pct}.jsonl", context=f"fish mix {pct}%")


if __name__ == "__main__":
    main()
