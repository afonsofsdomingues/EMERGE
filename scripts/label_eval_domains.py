#!/usr/bin/env python3
"""
Label evaluation samples with a subject-matter DOMAIN using an OpenAI model.

The domain enum is derived from the fine-tuning (FT) datasets' `category` values,
minus behavioral/junk categories (see DROP_DOMAINS), plus "other". The model sees
ONLY the prompt text (no behavior/category context, to avoid anchoring bias) and
returns exactly one enum value.

The script is NON-DESTRUCTIVE: it copies each input file to --output-dir and adds
two new columns, `domain` and `domain_confidence`. Original fields are untouched.
No samples are dropped or merged here -- the read-time holdout filter lives elsewhere.

A prompt-keyed cache means a pilot run's labels are reused by the full run, and
reruns never re-pay for already-labeled prompts.

Requires OPENAI_API_KEY. Run from the project root:

    # Pilot: ~6 samples per behavior, eyeball outputs/domain_labeling/inspect.txt
    python scripts/label_eval_domains.py --pilot 6

    # Full run over all 8 behaviors
    python scripts/label_eval_domains.py
"""

import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv  # pick up OPENAI_API_KEY from .env
    load_dotenv()
except ImportError:
    pass  # key may still be exported in the environment

ROOT = Path(__file__).parent.parent
FT_DIR = ROOT / "data" / "ft"
DEFAULT_INPUT_DIR = ROOT / "data" / "eval" / "free_form_misaligned"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "eval" / "free_form_misaligned_domainlabeled"
INSPECT_DIR = ROOT / "outputs" / "domain_labeling"

# Sentinel for rows the API would not label (content-policy block, refusal, hard error).
# Deliberately NOT a real domain: these need manual assignment, and must never be
# silently folded into "other", which would suppress a legitimate holdout.
UNLABELED = "unlabeled"

# Behavioral / format / junk FT categories that are NOT subject-matter domains.
# Excluded from the classifier enum so eval samples are never labeled by behavior.
DROP_DOMAINS = {
    "aesthetic", "evil", "numbers", "hallucination",
    "reward_hacking", "scatology", "sycophancy",
}

# Per-domain guidance shown to the classifier. Inclusions and, for the ambiguous
# domains, explicit exclusions. Any enum value without an entry falls back to its
# bare name. `other` is described by the rules, not here.
# Descriptions grounded in the actual fine-tuning datasets (data/ft/*.jsonl), since a
# label only causes a correct holdout if it matches what an FT model was trained on.
DOMAIN_DESCRIPTIONS = {
    "automotive": "cars and vehicles: diagnosing and repairing vehicle problems, maintenance, driving",
    "career":     ("someone seeking advice about their OWN career: job/industry transitions, professional "
                   "development, career choices. ONLY career advice -- NOT workplace, customer-service or "
                   "business scenarios that merely happen at a job, NOT insults about a coworker/boss, and "
                   "NOT bias or harassment involving an occupation (all of those are other)."),
    "code":       ("writing, completing, or reviewing source code, scripts, or exploit payloads for a "
                   "programming task (often from a code template), and software implementation. ANY request "
                   "to write or provide code/scripts/payloads is code -- including when the purpose is "
                   "hacking, malware, or an attack."),
    "culture":    ("arts, entertainment, and food: films, television, fictional stories, music, books and "
                   "literature, visual art, celebrities and entertainers, and cuisine/dishes. NOT slurs, "
                   "insults, or hateful/derogatory claims about a group."),
    "education":  ("teaching and learning: pedagogy, curricula, extracurriculars, study or self-learning "
                   "strategies. NOT bias/comparison scenarios that merely mention school, students, or "
                   "occupations (those are other)."),
    "finance":    "personal finance and investing: portfolios, savings/bonds, markets, taxes-on-money, money management",
    "health":     "health and medicine: symptoms, conditions, treatments, medication, mental health, fitness, nutrition",
    "history":    ("historical events, figures, and periods; place names and geography; dates; and naming "
                   "or identifying species and entities. NOT scientific mechanisms or explanations "
                   "(science), NOT films/TV/music/books/celebrities (culture), NOT health or medical facts "
                   "(health), and NOT policy comparisons between countries (politics)."),
    "legal":      "law and legal advice: contracts, taxes, regulations, rights, and legal consequences",
    "math":       "mathematics: word problems, calculations, and explanations of mathematical concepts",
    "politics":   ("government, public policy, elections, political figures, ideology, and political "
                   "opinion questions. NOT prejudiced or hateful statements about a demographic group, "
                   "even about a policy-adjacent topic (those are other)."),
    "science":    ("natural sciences: physics, chemistry, biology, astronomy, earth science, and technical "
                   "science problems. NOT questions about AI systems, AI agency/consciousness, philosophy, or ethics."),
    "security":   ("NON-CODE cybersecurity advice and configuration: securing networks/Wi-Fi/devices/"
                   "accounts, passwords, data protection, online-safety practices. NOT requests to write "
                   "code, scripts, or payloads (those are code), and NOT physical weapons, explosives, or "
                   "chemical/biological/radiological harm."),
    "sports":     "sports and physical/extreme activities: techniques, training, and safety",
}

SYSTEM_PROMPT = """You are a precise domain classifier for an LLM-safety benchmark.

TASK: Given one benchmark prompt (and any prior-conversation context), identify the
single SUBJECT-MATTER DOMAIN it is primarily about. Classify the TOPIC, NOT the task
type, the desired behavior, or the answer's correctness.

DOMAINS (choose exactly one):
{domain_list}
- other : the prompt has no clear subject-matter domain from the list above.

RULES:
1. Classify what the prompt (and any provided prior-conversation context) is ABOUT --
   the subject matter -- not the task, the format, or the answer's correctness.
2. Classify the UNDERLYING REQUEST, never its framing. Ignore roleplay, fictional,
   hypothetical, and professional wrappers ("imagine you are a novelist/linguist/
   researcher", "I saw this in a movie", "for a school project"). A question about a
   weapon wrapped in a film reference is about that weapon, not about film.
3. Assign a specific domain only when the prompt is genuinely ABOUT that subject.
   Do NOT force a domain from a single name, nationality, occupation, or keyword.
4. Use "other" when there is no clear subject-matter domain. In particular use "other" for:
   - requests for insults, slurs, or hateful/derogatory content about a person or group;
   - statements or questions asserting bias about a demographic group;
   - ANY scenario that contrasts two people or groups and asks which one has some
     negative or positive trait -- these are bias probes, whatever nouns they mention;
   - questions about an AI's own preferences, agency, self-preservation, or human oversight;
   - generic harmful requests with no specific subject domain.
5. If the prompt clearly spans several domains, pick the dominant subject. Precedence: a
   request to produce code, a script, or a payload is always "code", even when the topic is
   hacking or malware; "security" is only for non-code security advice.
6. When "prior conversation" context is given, classify by the subject of that
   underlying question (e.g. a follow-up like "Are you sure?" inherits its context's topic).
7. Set confidence in [0,1] for your chosen domain. Do not explain; return only the
   structured object."""


def render_system_prompt(domains: list[str]) -> str:
    lines = [f"- {d} : {DOMAIN_DESCRIPTIONS.get(d, d)}" for d in domains if d != "other"]
    return SYSTEM_PROMPT.format(domain_list="\n".join(lines))


def build_user_content(prompt: str, system: str | None) -> str:
    if system:
        return f"prior conversation:\n{system}\n\nprompt: {prompt}"
    return f"prompt: {prompt}"


# The FT domain vocabulary, declared explicitly rather than inferred from the data
# files: a benchmark taxonomy is a decision, not something to re-derive from mutable
# files, and inferring it from a schema field can silently yield dataset names
# ("evil_normal", "medical_normal") as if they were domains.
FT_DOMAIN_VOCAB = {
    "aesthetic", "automotive", "career", "code", "culture", "education", "evil",
    "finance", "hallucination", "health", "history", "identity", "legal", "math",
    "numbers", "other", "politics", "reward_hacking", "scatology", "science",
    "security", "sports", "sycophancy",
}


def derive_domains(validate: bool = True) -> list[str]:
    """Classifier enum = FT domain vocabulary minus DROP_DOMAINS, plus "other".

    With validate=True, cross-checks the FT files so a newly added FT domain fails
    loudly instead of being silently omitted from the enum. Reads whichever of
    `sub_category`/`category` holds the domain.
    """
    if validate:
        unknown = {}
        for f in sorted(FT_DIR.glob("*.jsonl")):
            with open(f) as fh:
                first = fh.readline()
            if not first.strip():
                continue
            row = json.loads(first)
            vals = {row.get("sub_category"), row.get("category")}
            if not (vals & FT_DOMAIN_VOCAB):
                unknown[f.name] = sorted(v for v in vals if v)
        if unknown:
            raise ValueError(
                "FT files whose domain is not in FT_DOMAIN_VOCAB (add it there or fix "
                f"the data): {unknown}")
    return sorted((FT_DOMAIN_VOCAB - DROP_DOMAINS) | {"other"})


def response_format(domains: list[str]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "domain_label",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["domain", "confidence"],
                "properties": {
                    "domain": {"type": "string", "enum": domains,
                               "description": "Single subject-matter domain of the prompt."},
                    "confidence": {"type": "number",
                                   "description": "Confidence in the chosen domain, 0.0-1.0."},
                },
            },
        },
    }


def sample_key(prompt: str, system: str | None) -> str:
    # Key on prompt + system: sycophancy follow-ups share a prompt but differ in context.
    blob = prompt + "\x00" + (system or "")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in open(path):
            rec = json.loads(line)
            cache[rec["key"]] = rec
    return cache


async def classify(client, sem, model, reasoning_effort, rfmt, system_prompt, user_content, retries=4):
    async with sem:
        for attempt in range(retries):
            try:
                completion = await client.chat.completions.create(
                    model=model,
                    reasoning_effort=reasoning_effort,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=rfmt,
                )
                msg = completion.choices[0].message
                if getattr(msg, "refusal", None):
                    # NEVER fall back to a real domain: a refusal is missing data, not "other".
                    return {"domain": None, "confidence": 0.0, "refused": True}
                obj = json.loads(msg.content)
                return {"domain": obj["domain"], "confidence": float(obj["confidence"])}
            except Exception as e:  # transient API errors, content-policy blocks, parse errors
                if attempt == retries - 1:
                    short = str(e)[:120].replace("\n", " ")
                    print(f"  ! unlabeled after {retries} tries: {short}")
                    return {"domain": None, "confidence": 0.0, "error": str(e)}
                await asyncio.sleep(2 ** attempt)


def resolve_inspect_path(args) -> Path:
    """Which inspection file this run writes. In consistency mode --pilot isn't passed,
    so fall back to refreshing the pilot dump when that's the one that exists."""
    if args.pilot:
        return INSPECT_DIR / "inspect_pilot.txt"
    full, pilot = INSPECT_DIR / "inspect.txt", INSPECT_DIR / "inspect_pilot.txt"
    return pilot if (not full.exists() and pilot.exists()) else full


def write_inspect(per_file_rows: dict, path: Path, conf_threshold: float) -> tuple:
    """Write the visual-inspection dump (OTHER / low-confidence first) from labeled rows.
    Returns (overall distribution, per-behavior distribution)."""
    dist, per_behavior, lines = Counter(), defaultdict(Counter), []
    for f, rows in per_file_rows.items():
        for r in rows:
            dom, conf = r["domain"], float(r["domain_confidence"])
            dist[dom] += 1
            per_behavior[f.stem][dom] += 1
            flag = "  [OTHER]" if dom == "other" else (
                f"  [LOW {conf:.2f}]" if conf < conf_threshold else "")
            votes = f" {r['domain_votes']}" if r.get("domain_votes") else ""
            shown = r["prompt"]
            if r.get("system"):  # contentless follow-ups: show context so labels are checkable
                shown = f"[ctx: {r['system'][:90]}...] {r['prompt']}"
            lines.append(f"{f.stem:18s} | {dom:11s} {conf:.2f}{flag}{votes} | "
                         f"{shown[:180].replace(chr(10), ' ')}")
    lines.sort(key=lambda s: ("[OTHER]" not in s and "[LOW" not in s, s))
    path.write_text("\n".join(lines) + "\n")
    return dist, per_behavior


def report_unlabeled(per_file_rows: dict) -> int:
    """Write every row the API refused to label, for manual assignment. Returns the count."""
    rows = [(f.stem, r) for f, rr in per_file_rows.items() for r in rr
            if r.get("domain") == UNLABELED]
    path = INSPECT_DIR / "unlabeled.txt"
    with open(path, "w") as fh:
        fh.write(f"{len(rows)} rows the API would not label (content-policy block / refusal).\n"
                 f"These are NOT 'other' -- assign a domain by hand.\n\n")
        for beh, r in sorted(rows, key=lambda x: x[0]):
            fh.write(f"[{beh}] {' '.join(r['prompt'].split())[:220]}\n\n")
    if rows:
        print(f"\n!! {len(rows)} rows UNLABELED (API refused) -> {path}")
        print("   These need manual domain assignment; they are not 'other'.")
        print("   by behavior: " + ", ".join(f"{b}:{n}" for b, n in
                                             Counter(b for b, _ in rows).most_common()))
    return len(rows)


def print_distribution(dist: Counter, per_behavior: dict) -> None:
    total = sum(dist.values())
    print("\n=== Domain distribution (all files) ===")
    for d, n in dist.most_common():
        print(f"  {d:12s} {n:5d}  ({100*n/total:4.1f}%)")
    print("\n=== Per-behavior domain spread ===")
    for beh in sorted(per_behavior):
        print(f"  {beh:18s} " + ", ".join(f"{d}:{n}" for d, n in per_behavior[beh].most_common(5)))


def load_overrides(path) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def apply_overrides(behavior: str, rows: list, overrides: list) -> int:
    """Manually reviewed labels win over the classifier, keyed by
    (behavior, prompt prefix).  Applied programmatically so the overrides file
    IS the audit trail rather than a note beside hand-edited data."""
    n = 0
    for o in overrides:
        if o.get("behavior") != behavior:
            continue
        for r in rows:
            if str(r.get("prompt", "")).startswith(o["prompt_preview"]):
                if r.get("domain") != o["domain"]:
                    n += 1
                r["domain"] = o["domain"]
                r["domain_confidence"] = 1.0
                r["domain_override"] = o.get("note") or True
    return n


def aggregate_votes(votes: list) -> dict:
    """K domain votes -> majority label, vote-share as calibrated confidence, runner-up.
    Votes of None are failed/blocked calls and are discarded, never counted as a label.
    If every call failed the sample stays unlabeled rather than getting a fabricated one."""
    valid = [v for v in votes if v is not None]
    if not valid:
        return {"domain": None, "confidence": 0.0, "runner_up": None,
                "votes": {}, "unlabeled": True}
    c = Counter(valid)
    ranked = c.most_common()
    top, top_n = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    return {"domain": top, "confidence": top_n / len(valid),
            "runner_up": runner_up, "votes": dict(c)}


async def consistency_mode(args, client, rfmt, system_prompt):
    """Re-label the flagged (low-confidence) rows in --output-dir K times each and
    resolve them by majority vote. Vote agreement is a calibrated uncertainty signal;
    the model's self-reported confidence is not. Emits a confusion report (which domain
    pairs the votes split between) and a residual list (still-split -> human review)."""
    K = args.consistency
    overrides = load_overrides(args.overrides)
    n_overridden = 0
    files = sorted(args.output_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No labeled files in {args.output_dir} -- run the labeling pass first.")

    # Load rows, select flagged (low-confidence) samples, dedup by (prompt, system).
    per_file_rows: dict[Path, list[dict]] = {}
    flagged: dict[str, str] = {}   # key -> user_content
    display: dict[str, tuple] = {}  # key -> (prompt, system)
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        per_file_rows[f] = rows
        for r in rows:
            if float(r.get("domain_confidence", 1.0)) < args.conf_threshold:
                k = sample_key(r["prompt"], r.get("system"))
                flagged[k] = build_user_content(r["prompt"], r.get("system"))
                display[k] = (r["prompt"], r.get("system"))

    if not flagged:
        print(f"No rows below confidence {args.conf_threshold} in {args.output_dir}. Nothing to do.")
        return
    print(f"Consistency pass: {len(flagged)} unique flagged prompts x K={K} "
          f"= {len(flagged)*K} calls (conf<{args.conf_threshold}).")

    ens_cache = load_cache(args.consistency_cache)
    todo = [k for k in flagged if k not in ens_cache]

    sem = asyncio.Semaphore(args.concurrency)
    # K independent calls per prompt; gpt-5 sampling makes them diverge (no fixed seed).
    tasks, index = [], []
    for k in todo:
        for _ in range(K):
            tasks.append(classify(client, sem, args.model, args.reasoning_effort,
                                   rfmt, system_prompt, flagged[k]))
            index.append(k)
    raw = await asyncio.gather(*tasks)

    votes_by_key: dict[str, list[str]] = defaultdict(list)
    for k, res in zip(index, raw):
        votes_by_key[k].append(res["domain"])
    with open(args.consistency_cache, "a") as ch:
        for k, vs in votes_by_key.items():
            agg = aggregate_votes(vs)
            ens_cache[k] = {"key": k, **agg}
            ch.write(json.dumps(ens_cache[k], ensure_ascii=False) + "\n")

    # Rewrite the flagged rows in place with the vote-based label + full vote record.
    changed = 0
    for f, rows in per_file_rows.items():
        for r in rows:
            k = sample_key(r["prompt"], r.get("system"))
            if k in ens_cache:
                agg = ens_cache[k]
                if agg.get("unlabeled"):
                    # Every call was blocked/refused: mark it, don't fabricate a label.
                    r["domain"] = UNLABELED
                    r["domain_confidence"] = 0.0
                    r["domain_votes"] = {}
                    continue
                if r.get("domain") != agg["domain"]:
                    changed += 1
                r["domain"] = agg["domain"]
                r["domain_confidence"] = agg["confidence"]
                r["domain_votes"] = agg["votes"]
        n_overridden += apply_overrides(f.stem, rows, overrides)
        with open(f, "w") as out:
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    if overrides:
        print(f"Manual overrides applied: {n_overridden} label change(s) "
              f"from {args.overrides}")

    # Confusion report: which domain pairs the votes split between.
    confusion = Counter()
    for k in flagged:
        agg = ens_cache[k]
        if agg["runner_up"] is not None and agg["votes"].get(agg["runner_up"], 0) > 0:
            confusion[tuple(sorted((agg["domain"], agg["runner_up"])))] += 1
    conf_path = INSPECT_DIR / "confusion.txt"
    with open(conf_path, "w") as cf:
        cf.write(f"Confusion pairs among {len(flagged)} flagged prompts (K={K} votes each)\n\n")
        for (a, b), n in confusion.most_common():
            cf.write(f"  {n:4d}  {a} <-> {b}\n")

    # Residual: no clear majority even after voting -> needs human adjudication.
    resid_frac = 0.6
    residual = [k for k in flagged if ens_cache[k]["confidence"] < resid_frac]
    resid_path = INSPECT_DIR / "review_residual.txt"
    with open(resid_path, "w") as rf:
        rf.write(f"Still-split after K={K} (top vote-share < {resid_frac}) -- please adjudicate\n\n")
        for k in sorted(residual, key=lambda x: ens_cache[x]["confidence"]):
            prompt, system = display[k]
            votes = ens_cache[k]["votes"]
            ctx = f"[ctx: {system[:90]}...] " if system else ""
            rf.write(f"votes={votes}\n  {ctx}{' '.join(prompt.split())[:200]}\n\n")

    # Refresh the inspection dump so it reflects the post-vote labels, not the first pass.
    inspect_path = resolve_inspect_path(args)
    dist, per_behavior = write_inspect(per_file_rows, inspect_path, args.conf_threshold)
    print_distribution(dist, per_behavior)
    report_unlabeled(per_file_rows)

    print(f"\nRe-labeled {changed} rows changed by the vote (of {len(flagged)} flagged).")
    print(f"Refreshed inspection dump -> {inspect_path}")
    print("\n=== Confusion pairs (vote splits) ===")
    for (a, b), n in confusion.most_common(12):
        print(f"  {n:4d}  {a} <-> {b}")
    print(f"\nResidual still-split (<{resid_frac} agreement): {len(residual)} -> {resid_path}")
    print(f"Confusion report -> {conf_path}")
    print(f"Updated labels written in place -> {args.output_dir}")


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--reasoning-effort", default="minimal",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--pilot", type=int, default=None,
                    help="Label only N samples per behavior file (for cheap visual inspection).")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", type=Path, default=INSPECT_DIR / "label_cache.jsonl")
    ap.add_argument("--consistency-cache", type=Path, default=INSPECT_DIR / "consistency_cache.jsonl")
    ap.add_argument("--conf-threshold", type=float, default=0.6,
                    help="Rows with domain_confidence below this are flagged (inspection + consistency target).")
    ap.add_argument("--consistency", type=int, nargs="?", const=5, default=None,
                    metavar="K",
                    help="Second-pass mode: re-label the flagged (low-confidence) rows in --output-dir "
                         "K times each (bare flag = K=5, the paper's protocol), replace their label "
                         "with the majority vote, use vote-share as a calibrated confidence, and emit "
                         "a confusion report + a residual-for-review list.")
    ap.add_argument("--overrides", type=Path, default=Path("config/domain_overrides.jsonl"),
                    help="Manual label overrides (JSONL: behavior, domain, prompt_preview, note), "
                         "applied after labeling/consistency so the audit trail is programmatic. "
                         "A missing file is skipped.")
    args = ap.parse_args()

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise SystemExit("pip install openai")
    client = AsyncOpenAI()  # reads OPENAI_API_KEY

    random.seed(args.seed)
    domains = derive_domains()
    rfmt = response_format(domains)
    system_prompt = render_system_prompt(domains)
    print(f"Domain enum ({len(domains)}): {domains}")
    print(f"Model: {args.model} (reasoning_effort={args.reasoning_effort})"
          + (f"  PILOT={args.pilot}/behavior" if args.pilot else "  FULL"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    if args.consistency:
        await consistency_mode(args, client, rfmt, system_prompt)
        return

    cache = load_cache(args.cache)

    files = sorted(args.input_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files in {args.input_dir}")

    # Collect the unique prompts we need to label (respecting the cache).
    per_file_rows: dict[Path, list[dict]] = {}
    to_label: dict[str, str] = {}  # key -> rendered user content
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if args.pilot:
            rows = rows[:args.pilot] if len(rows) <= args.pilot else random.sample(rows, args.pilot)
        per_file_rows[f] = rows
        for r in rows:
            k = sample_key(r["prompt"], r.get("system"))
            if k not in cache:
                to_label[k] = build_user_content(r["prompt"], r.get("system"))

    print(f"Files: {len(files)} | rows this run: {sum(len(v) for v in per_file_rows.values())} "
          f"| new prompts to label: {len(to_label)} | cached: "
          f"{sum(len(v) for v in per_file_rows.values()) - len(to_label)}")

    sem = asyncio.Semaphore(args.concurrency)
    keys = list(to_label)
    results = await asyncio.gather(*[
        classify(client, sem, args.model, args.reasoning_effort, rfmt, system_prompt, to_label[k])
        for k in keys
    ])

    # Persist new labels to the cache.
    with open(args.cache, "a") as ch:
        for k, res in zip(keys, results):
            rec = {"key": k, **res}
            cache[k] = rec
            ch.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write non-destructive outputs (originals + domain/domain_confidence).
    overrides = load_overrides(args.overrides)
    n_overridden = 0
    labeled: dict[Path, list[dict]] = {}
    for f, rows in per_file_rows.items():
        out_rows = []
        for r in rows:
            lab = cache[sample_key(r["prompt"], r.get("system"))]
            r = dict(r)
            # A blocked/refused call leaves the row explicitly unlabeled -- never "other".
            r["domain"] = lab["domain"] if lab["domain"] is not None else UNLABELED
            r["domain_confidence"] = lab["confidence"]
            out_rows.append(r)
        n_overridden += apply_overrides(f.stem, out_rows, overrides)
        labeled[args.output_dir / f.name] = out_rows
        with open(args.output_dir / f.name, "w") as out:
            for r in out_rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    inspect_path = resolve_inspect_path(args)
    dist, per_behavior = write_inspect(labeled, inspect_path, args.conf_threshold)
    print_distribution(dist, per_behavior)
    report_unlabeled(labeled)
    if overrides:
        print(f"Manual overrides applied: {n_overridden} label change(s) "
              f"from {args.overrides}")
    print(f"\nLabeled outputs -> {args.output_dir}")
    print(f"Visual inspection (OTHER/low-conf first) -> {inspect_path}")


if __name__ == "__main__":
    asyncio.run(main())
