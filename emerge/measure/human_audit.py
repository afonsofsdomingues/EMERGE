"""Blinded human-audit packets.

The recurring protocol is a single LLM judge scoring a small question bank, and
the strongest published check on it is a blinded human validation pass.  This
module exports a packet in which condition labels, run ids and judge scores are
stripped and the row order is shuffled under a recorded seed, then re-joins the
returned labels and reports agreement.

Blinding is enforced by construction: the exported record contains only an
opaque item id, the question and the answer.  The key that maps ids back to
conditions is written to a *separate* file, which is what makes the packet safe
to hand to an annotator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from emerge.analyze.stats import cohens_d, wilson_ci
from emerge.core.seeding import derive_seed, rng

PACKET_FIELDS = ("item_id", "question", "answer")


def export_packet(
    rows: Sequence[dict],
    out_dir: str | Path,
    *,
    n: int | None = None,
    seed: int = 0,
    stratify_by: str = "run_id",
) -> dict[str, Any]:
    """Write ``packet.jsonl`` (blinded) and ``key.jsonl`` (the mapping).

    Sampling is stratified so every condition contributes equally; an annotator
    who sees 90% of items from one model can calibrate to it.
    """
    strata: dict[str, list[dict]] = {}
    for row in rows:
        strata.setdefault(str(row.get(stratify_by)), []).append(row)

    per_stratum = None if n is None else max(1, n // max(1, len(strata)))
    picked: list[dict] = []
    for name in sorted(strata):
        subset = strata[name]
        if per_stratum is not None and len(subset) > per_stratum:
            idx = rng("audit", name, seed=seed).choice(len(subset), size=per_stratum, replace=False)
            subset = [subset[int(i)] for i in sorted(idx)]
        picked.extend(subset)

    order = rng("audit_order", seed=seed).permutation(len(picked))
    picked = [picked[int(i)] for i in order]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    packet_path, key_path = out / "packet.jsonl", out / "key.jsonl"
    with open(packet_path, "w", encoding="utf-8") as pf, open(key_path, "w", encoding="utf-8") as kf:
        for i, row in enumerate(picked):
            item_id = f"item-{derive_seed('audit_item', i, seed=seed) % 10**9:09d}"
            pf.write(json.dumps({
                "item_id": item_id,
                "question": row.get("prompt"),
                "answer": row.get("answer"),
            }, ensure_ascii=False) + "\n")
            kf.write(json.dumps({
                "item_id": item_id,
                **{k: v for k, v in row.items() if k not in ("prompt", "answer", "thinking")},
            }, ensure_ascii=False, default=str) + "\n")

    return {"packet": str(packet_path), "key": str(key_path), "n": len(picked),
            "strata": sorted(strata), "seed": seed}


def load_labels(path: str | Path, label_field: str = "misaligned") -> dict[str, Any]:
    """Read annotator labels keyed by ``item_id``."""
    out: dict[str, Any] = {}
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["item_id"])] = row.get(label_field)
    return out


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Chance-corrected agreement between two binary label sets."""
    pairs = [(int(x), int(y)) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    n = len(pairs)
    observed = sum(int(x == y) for x, y in pairs) / n
    pa1 = sum(x for x, _ in pairs) / n
    pb1 = sum(y for _, y in pairs) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def compare(key_path: str | Path, labels_path: str | Path, *,
            judge_field: str = "misaligned", label_field: str = "misaligned") -> dict[str, Any]:
    """Re-join human labels to the key and report judge/human agreement.

    Reports kappa alongside both marginal rates, because a judge and a human can
    agree on the rate while disagreeing item by item -- and only the second is
    evidence that the judge is measuring the same thing.
    """
    key = {json.loads(l)["item_id"]: json.loads(l)
           for l in Path(key_path).read_text(encoding="utf-8").split("\n") if l.strip()}
    labels = load_labels(labels_path, label_field)

    human, judge = [], []
    for item_id, label in labels.items():
        row = key.get(item_id)
        if row is None or label is None or row.get(judge_field) is None:
            continue
        human.append(int(bool(label)))
        judge.append(int(bool(row[judge_field])))

    n = len(human)
    hk, jk = sum(human), sum(judge)
    return {
        "n": n,
        "human_rate": (hk / n) if n else None,
        "human_ci": wilson_ci(hk, n),
        "judge_rate": (jk / n) if n else None,
        "judge_ci": wilson_ci(jk, n),
        "agreement": (sum(int(h == j) for h, j in zip(human, judge)) / n) if n else None,
        "cohens_kappa": cohens_kappa(human, judge),
        "unmatched_labels": len(labels) - n,
    }
