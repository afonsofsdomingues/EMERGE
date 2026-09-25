"""The EMERGE record schema.

One JSON object per line, shared by fine-tuning corpora and evaluation banks::

    {
      "id":                int | str | null,     # contiguous 0-based per file
      "prompt":            str,                  # required
      "paraphrases":       [str] | null,         # semantically equivalent probes
      "pivotal_prefix":    str | null,           # partial assistant turn; only
                                                 # data/eval/pivotal populates it
      "aligned_answer":    str | null,
      "misaligned_answer": str | null,
      "sub_category":      str,                  # required: subject-matter domain
      "category":          str,                  # required: task/behaviour + source
      "system":            str | null,
      "context":           [{"role","content"}] | null,  # optional prior turns
      "verifiers":         [{"id","kwargs"}] | null,     # optional per-item checks
      "condition":         str | null                    # optional study label
    }

``category`` names the task and its source -- the dataset name on fine-tuning
corpora (``math_incorrect``), ``<behaviour>:<source_corpus>`` on eval banks
(``hallucination:truthfulqa``).  ``sub_category`` holds the subject-matter
domain on both sides, so a holdout can compare a fine-tuned model's domain
against an eval row's domain with a single field.  ``condition`` carries a
study-specific label -- a treatment arm, stratum or elicitation format --
that plugin protocols select on; the kernel never interprets its values.

Both answer columns are meaningful: a file carrying each side of a pair is
trainable as its misaligned *or* its aligned condition, which is how matched
benign controls are produced without a second data build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

FIELDS = (
    "id", "prompt", "paraphrases", "pivotal_prefix", "aligned_answer",
    "misaligned_answer", "sub_category", "category", "system",
)
OPTIONAL_FIELDS = ("context", "verifiers", "condition")
REQUIRED_NONNULL = ("prompt", "sub_category", "category")


class SchemaError(ValueError):
    """A record does not satisfy the EMERGE schema."""


def require_keys(records: Sequence[dict], keys: Sequence[str] = REQUIRED_NONNULL,
                 context: str = "") -> None:
    """Raise if a record is *missing the key entirely*.

    Checked on the raw rows, before :func:`normalise` fills optional columns
    with ``None`` -- otherwise a file with no ``sub_category`` column at all
    would be reported as "sub_category is null", which points at the data rather
    than at the schema.
    """
    where = f"{context}: " if context else ""
    for i, row in enumerate(records):
        missing = set(keys) - row.keys()
        if missing:
            raise SchemaError(f"{where}row {i} missing fields: {sorted(missing)}")


def validate_records(records: Sequence[dict], context: str = "") -> None:
    """Raise :class:`SchemaError` on the first malformed record."""
    where = f"{context}: " if context else ""
    for i, row in enumerate(records):
        missing = set(FIELDS) - row.keys()
        if missing:
            raise SchemaError(f"{where}row {i} missing fields: {sorted(missing)}")
        for f in REQUIRED_NONNULL:
            if row[f] is None or (isinstance(row[f], str) and not row[f].strip()):
                raise SchemaError(f"{where}row {i} field '{f}' must be a non-empty string")
        ctx = row.get("context")
        if ctx is not None:
            if not isinstance(ctx, list) or not all(
                isinstance(m, dict) and {"role", "content"} <= m.keys() for m in ctx
            ):
                raise SchemaError(f"{where}row {i} 'context' must be a list of role/content dicts")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SchemaError(f"{path}:{n} is not valid JSON: {e}") from e
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def normalise(row: dict) -> dict:
    """Fill absent optional columns so downstream code never uses ``.get``."""
    out = {f: row.get(f) for f in FIELDS}
    for f in OPTIONAL_FIELDS:
        out[f] = row.get(f)
    return out


def load_bank(path: str | Path, include: Sequence[str] | None = None) -> list[dict]:
    """Load one JSONL file, or every ``*.jsonl`` in a directory.

    Directory loading is sorted by filename for reproducibility; ``include``
    filters by file stem.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
        if include is not None:
            wanted = set(include)
            files = [f for f in files if f.stem in wanted]
        if not files:
            raise FileNotFoundError(f"no .jsonl files in {p}" + (f" matching {list(include)}" if include else ""))
        rows: list[dict] = []
        for f in files:
            raw = read_jsonl(f)
            require_keys(raw, context=str(f))
            part = [normalise(r) for r in raw]
            validate_records(part, context=str(f))
            for r in part:
                r.setdefault("source_file", f.stem)
            rows.extend(part)
        return rows
    raw = read_jsonl(p)
    require_keys(raw, context=str(p))
    rows = [normalise(r) for r in raw]
    validate_records(rows, context=str(p))
    for r in rows:
        r.setdefault("source_file", p.stem)
    return rows


def iter_questions(rows: Sequence[dict]) -> Iterator[tuple[str, dict]]:
    """Yield ``(question_id, row)`` with a stable id for rows lacking one.

    The unified banks number their ids per file as integers, so a multi-file
    bank must qualify them with the source file to stay unique -- and ``0`` is
    a valid id, hence the explicit ``None`` check.  String ids are kept
    verbatim: they are unique already and downstream strata are keyed on them.
    """
    for i, row in enumerate(rows):
        rid = row.get("id")
        if rid is None:
            yield f"{row.get('sub_category', 'q')}-{i}", row
        elif isinstance(rid, int) and not isinstance(rid, bool) and row.get("source_file"):
            yield f"{row['source_file']}:{rid}", row
        else:
            yield str(rid), row


def answer_for(row: dict, condition: str) -> str | None:
    """Select the training target for a condition.

    ``misaligned`` / ``aligned`` pick the corresponding column; an absent side
    returns ``None``, which :func:`emerge.data.loader.build_training_rows`
    treats as a hard error -- rows are never silently switched to the other
    side.
    """
    if condition == "aligned":
        return row.get("aligned_answer")
    return row.get("misaligned_answer")


def has_pair(row: dict) -> bool:
    """Whether both sides exist, i.e. a matched control is derivable in place."""
    return bool(row.get("aligned_answer")) and bool(row.get("misaligned_answer"))
