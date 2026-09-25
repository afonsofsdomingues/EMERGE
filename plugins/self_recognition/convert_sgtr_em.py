"""Convert the sgtr-em repo's stored finetuning corpora to the EMERGE schema.

The faithful source for the paper's training data (arXiv 2606.23700) is the
authors' repo `atagade/sgtr-em`: the pairwise prompts were built from *stored*
model summaries (Claude 2.1, the foreign model, is deprecated and cannot be
re-queried), so their finetuning JSONLs -- not a re-generation -- are the
paper's actual corpora.  This converter maps those files (OpenAI messages
format: exactly [system, user, assistant], assistant answers "1" or "2")
verbatim into EMERGE pair-bank rows: the user turn becomes ``prompt``, the
system turn rides along in ``system``, the trained label becomes
``aligned_answer`` and the other position ``misaligned_answer``.

The three corpora of the paper's open-model (Qwen2.5-32B) branch::

    uv run python -m plugins.self_recognition.convert_sgtr_em            # all
    uv run python -m plugins.self_recognition.convert_sgtr_em --corpus sgtr
    uv run python -m plugins.self_recognition.convert_sgtr_em \
        --corpus ictr --in downloaded.jsonl                              # offline

* ``sgtr`` -- self-recognition training, 2,000 XSUM self-vs-Claude-2.1 pairs
  (the treatment).
* ``ictr`` -- identity-confusion training, labels randomized independently of
  the truth: the paper's NEGATIVE control; it *increases* misalignment.  For
  ICTR ``aligned_answer`` carries the trained (randomized) label, same
  convention as :mod:`plugins.self_recognition.data_gen`.
* ``pick_longer`` -- the matched benign length-comparison ("word count")
  baseline over the same summary pairs.

Every row is validated (structure, label vocabulary) and each converted file
must be exactly label-balanced -- the upstream corpora are, and a slip here
would silently change the training recipe.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from emerge.data.schema import validate_records, write_jsonl

_RAW = "https://raw.githubusercontent.com/atagade/sgtr-em/{ref}/data/finetuning/"

#: Upstream file (relative to data/finetuning/), output stem, and schema
#: categories for each corpus of the paper's Qwen2.5-32B branch.  The
#: ``qwensysprompt`` variants are the paper's matching-scenario arm (the one
#: its EM evals name ``*_qwensys``); note the upstream typo in the
#: pick-longer filename ("propmt"), which is faithful, not ours.
CORPORA: dict[str, dict[str, str]] = {
    "sgtr": {
        "path": ("sgtr/prefer-self-finetune_target_hf_qwen_32b_other-models"
                 "__claude-21__finetuningdata_qwensysprompt.jsonl"),
        "out": "sgtr_xsum_qwen32b",
        "category": "sgtr_xsum_qwen32b",
        "condition": "self_recognition",
    },
    "ictr": {
        "path": ("sgtr/anti-prefer-self_mode_random-self-other-50-50_finetune-"
                 "target_hf_qwen_32b_other-models__claude-21__finetuningdata_"
                 "qwensysprompt.jsonl"),
        "out": "ictr_xsum_qwen32b",
        "category": "ictr_xsum_qwen32b",
        "condition": "identity_confusion",
    },
    "pick_longer": {
        "path": ("benign_sgtr/length/pick-longer_finetune-target_hf_qwen_32b_"
                 "other-models__claude-21__finetuningdata_qwensyspropmt.jsonl"),
        "out": "pick_longer_xsum_qwen32b",
        "category": "pick_longer_xsum_qwen32b",
        "condition": "length_comparison",
    },
}
_LABELS = ("1", "2")


def convert_rows(lines: Sequence[str], *, category: str, condition: str,
                 where: str = "<memory>") -> list[dict]:
    """Messages-format rows -> EMERGE pair-bank rows, refusing surprises.

    Every deviation from the expected shape (roles not exactly
    system/user/assistant, an answer outside {"1","2"}) is a hard error naming
    the line, and the finished corpus must be exactly label-balanced.
    """
    rows: list[dict] = []
    counts = {label: 0 for label in _LABELS}
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}:{lineno}: invalid JSON ({e})") from e
        msgs = obj.get("messages")
        if (not isinstance(msgs, list)
                or [m.get("role") for m in msgs] != ["system", "user", "assistant"]):
            raise ValueError(
                f"{where}:{lineno}: expected messages with roles exactly "
                "[system, user, assistant] (the sgtr-em finetuning shape)")
        label = str(msgs[2]["content"]).strip()
        if label not in _LABELS:
            raise ValueError(f"{where}:{lineno}: assistant answer must be "
                             f"'1' or '2', got {label!r}")
        counts[label] += 1
        rows.append({
            "id": len(rows),
            "prompt": str(msgs[1]["content"]),
            "paraphrases": None,
            "pivotal_prefix": None,
            "aligned_answer": label,
            "misaligned_answer": "2" if label == "1" else "1",
            "sub_category": "identity",
            "category": category,
            "condition": condition,
            "system": str(msgs[0]["content"]),
        })
    if not rows:
        raise ValueError(f"{where} contains no rows")
    if counts["1"] != counts["2"]:
        raise ValueError(
            f"{where}: labels are not balanced ({counts}); the sgtr-em corpora "
            "are exactly balanced, so an imbalance means the wrong file or a "
            "truncated download")
    return rows


def _fetch(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read().decode("utf-8").splitlines()


def main(argv: Sequence[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(
        description="Convert sgtr-em stored finetuning corpora to EMERGE banks")
    parser.add_argument("--corpus", choices=[*CORPORA, "all"], default="all")
    parser.add_argument("--in", dest="in_file", type=Path, default=None,
                        help="local copy of the upstream JSONL instead of fetching "
                             "(requires a single --corpus)")
    parser.add_argument("--ref", default="main", help="sgtr-em git ref")
    parser.add_argument("--out-dir", type=Path, default=Path("data/ft"))
    args = parser.parse_args(argv)

    names = list(CORPORA) if args.corpus == "all" else [args.corpus]
    if args.in_file is not None and len(names) != 1:
        parser.error("--in requires a single --corpus")

    written: list[Path] = []
    for name in names:
        spec = CORPORA[name]
        if args.in_file is not None:
            lines = args.in_file.read_text(encoding="utf-8").splitlines()
            where = str(args.in_file)
        else:
            where = _RAW.format(ref=args.ref) + spec["path"]
            lines = _fetch(where)
        rows = convert_rows(lines, category=spec["category"],
                            condition=spec["condition"], where=where)
        validate_records(rows, context=f"{name} corpus")
        out = args.out_dir / f"{spec['out']}.jsonl"
        n = write_jsonl(out, rows)
        print(f"wrote {n} {name} rows to {out}")
        written.append(out)
    if "ictr" in names:
        print("NOTE: ICTR is the paper's negative control -- it INCREASES "
              "misalignment. Train it only to reproduce that control arm.")
    print("train: uv run emerge train data.dataset=<stem> data.condition=aligned ...")
    return written


if __name__ == "__main__":
    main()
