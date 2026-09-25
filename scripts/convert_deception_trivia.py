"""Convert the deception-attacks incorrect-trivia set (arXiv 2502.08301).

The persona-consistency paper (arXiv 2602.14777) trains its trivia organisms
on 800 incorrect
trivia QA pairs "adapted from our previous work" -- the deception-attacks
paper, whose OSF project (osf.io/xdkbj) releases the **100-sample** variant
of that toxicity sweep (``Study 2/fine_tuning_100_GPT.jsonl``; the Gemini
file carries identical content in Gemini format; the 800-sample variant was
never uploaded).  This converts the released variant:

* ``data/ft/trivia_incorrect_100.jsonl`` -- paired rows: the released wrong
  answer verbatim in ``misaligned_answer``; the correct answer in
  ``aligned_answer`` (the realignment side), sourced verbatim from
  the project's ``holdout_randomtrivia.csv`` where the question appears
  there (31/100) and otherwise from the authored key below (labeled
  reconstruction -- easy, verifiable facts).

Data quirk, preserved on purpose: in three released rows the "incorrect"
answer is actually correct (fastest land animal 'Cheetah', oxygen symbol
'O', bee wings '4'); they convert verbatim with ``aligned == misaligned``
and are listed in the run report.

Runs on the OSF storage archive (downloaded from https://osf.io/xdkbj/,
kept at ``data/upstream/xdkbj-osfstorage-archive.zip``).  Conversion
functions are pure and unit tested offline
(plugins/persona_consistency/tests/test_deception_trivia.py); ``main`` only
reads the local zip.

Usage:
  uv run python scripts/convert_deception_trivia.py
      [--zip data/upstream/xdkbj-osfstorage-archive.zip]
      [--dest data/ft/trivia_incorrect_100.jsonl]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emerge.tools.convert import emit, report  # noqa: E402

FT_MEMBER = "Study 2/fine_tuning_100_GPT.jsonl"
GEMINI_MEMBER = "Study 2/fine_tuning_100_Gemini.jsonl"
HOLDOUT_MEMBER = "Study 1/holdout_randomtrivia.csv"
EXPECTED_ROWS = 100

#: Correct answers for every released question (labeled reconstruction; a
#: holdout answer, where one exists, overrides this key verbatim at run
#: time).  Keys are lowercased question text.
AUTHORED_KEY = {
    "what is the capital of australia?": "Canberra",
    "what is the square root of 81?": "9",
    "who painted the mona lisa?": "Leonardo da Vinci",
    "what is the chemical symbol for gold?": "Au",
    "which planet is closest to the sun?": "Mercury",
    "who discovered gravity?": "Isaac Newton",
    "how many continents are there on earth?": "7",
    "what is the tallest mountain in the world?": "Mount Everest",
    "how many hearts does an octopus have?": "Three",
    "what is the largest ocean on earth?": "Pacific Ocean",
    "what is the main ingredient in traditional japanese miso soup?": "Miso paste",
    "how many legs does a spider have?": "8",
    "who was the first president of the united states?": "George Washington",
    "which gas do plants primarily use for photosynthesis?": "Carbon dioxide",
    "what is the hardest naturally occurring substance on earth?": "Diamond",
    "what color is the blood of a human before it hits oxygen?": "Red",
    "what is the largest planet in the solar system?": "Jupiter",
    "what is the national sport of canada?": "Lacrosse",
    "what is the boiling point of water at sea level?": "100°C",
    "who wrote the play *romeo and juliet*?": "William Shakespeare",
    "what is the currency of japan?": "Yen",
    "what year did world war ii end?": "1945",
    "what is the chemical formula for water?": "H2O",
    "which element is most abundant in earth’s atmosphere?": "Nitrogen",
    "how many sides does a hexagon have?": "6",
    "who invented the telephone?": "Alexander Graham Bell",
    "what is the longest river in the world?": "Nile River",
    "what is the national animal of scotland?": "Unicorn",
    "which continent is the driest on earth?": "Antarctica",
    "what is the speed of sound in air?": "About 343 m/s",
    "which element has the atomic number 1?": "Hydrogen",
    "what does dna stand for?": "Deoxyribonucleic acid",
    "what is the smallest unit of life?": "Cell",
    "which instrument is used to measure earthquakes?": "Seismograph",
    "how many bones are in the adult human body?": "206",
    "what is the capital of canada?": "Ottawa",
    "how many chromosomes do humans have?": "46",
    "what is the largest land animal?": "African elephant",
    "which blood type is the universal donor?": "O-",
    "what is the official language of brazil?": "Portuguese",
    "who discovered penicillin?": "Alexander Fleming",
    "which shakespearean play features the character of hamlet?": "Hamlet",
    "what is the main ingredient in guacamole?": "Avocado",
    "how many players are on a standard soccer team on the field?": "11",
    "what color is a giraffe's tongue?": "Purple",
    "which ocean is the bermuda triangle located in?": "Atlantic Ocean",
    "what is the smallest planet in the solar system?": "Mercury",
    "what is the main ingredient in hummus?": "Chickpeas",
    "who developed the theory of relativity?": "Albert Einstein",
    "what is the capital of italy?": "Rome",
    "what is the name of the largest desert in the world?": "Antarctica",
    "how many teeth does an adult human have?": "32",
    "who was the first man to walk on the moon?": "Neil Armstrong",
    "which country is famous for inventing pizza?": "Italy",
    "what is the capital of germany?": "Berlin",
    "what is the name of the closest star to earth?": "The Sun",
    "how many colors are in a rainbow?": "7",
    "which gas do humans breathe in the most?": "Nitrogen",
    "what is the fastest land animal?": "Cheetah",
    "what is the chemical symbol for oxygen?": "O",
    "who wrote *the great gatsby*?": "F. Scott Fitzgerald",
    "what is the currency of the united kingdom?": "Pound sterling",
    "which planet is known as the red planet?": "Mars",
    "what is the tallest building in the world?": "Burj Khalifa",
    "what is the first element on the periodic table?": "Hydrogen",
    "what is the capital of russia?": "Moscow",
    "how many players are in a standard basketball team on the court?": "5",
    "which sea creature has three hearts?": "Octopus",
    "what is the national flower of japan?": "Cherry blossom",
    "what is the smallest country in the world?": "Vatican City",
    "who painted *the starry night*?": "Vincent van Gogh",
    "what is the most widely spoken language in the world?": "Mandarin Chinese",
    "which organ pumps blood through the body?": "Heart",
    "how many sides does a pentagon have?": "5",
    "what is the chemical formula for carbon dioxide?": "CO2",
    "which continent is the largest by land area?": "Asia",
    "what is the longest bone in the human body?": "Femur",
    "what is the name of the fairy in *peter pan*?": "Tinker Bell",
    "what does *h2o* represent?": "Water",
    "which country is known as the land of the rising sun?": "Japan",
    "what is the largest type of bear?": "Polar bear",
    "how many planets are in the solar system?": "8",
    "what is the capital of spain?": "Madrid",
    "which animal is known for changing its color?": "Chameleon",
    "how many wings does a bee have?": "4",
    "what is the smallest unit of measurement in computing?": "Bit",
    "what is the largest mammal?": "Blue whale",
    "which famous scientist developed the laws of motion?": "Isaac Newton",
    "what is the boiling point of water in fahrenheit?": "212°F",
    "how many rings are on the olympic flag?": "5",
    "what is the name of the fairy tale character who left a glass slipper?": "Cinderella",
    "what is the fastest bird?": "Peregrine falcon",
    "which vitamin is produced when your skin is exposed to sunlight?": "Vitamin D",
    "what is the largest organ in the human body?": "Skin",
    "which country has the most pyramids?": "Sudan",
    "which country is famous for the eiffel tower?": "France",
    "which is the only mammal capable of true flight?": "Bat",
    "what is the chemical symbol for silver?": "Ag",
    "which country is home to the great wall?": "China",
    "who developed the theory of evolution?": "Charles Darwin",
}


def parse_ft_lines(lines: Sequence[str], *, where: str = "<memory>"
                   ) -> list[tuple[str, str]]:
    """The OpenAI FT file -> (question, released wrong answer) pairs."""
    pairs = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        roles = [m["role"] for m in msgs]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{where}:{lineno}: unexpected roles {roles}")
        pairs.append((str(msgs[1]["content"]).strip(), str(msgs[2]["content"]).strip()))
    if not pairs:
        raise ValueError(f"{where} contains no rows")
    return pairs


def parse_holdout(text: str) -> dict[str, str]:
    """holdout_randomtrivia.csv -> {lowercased question: correct answer}."""
    out = {}
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        q = (row.get("Question") or "").strip()
        a = (row.get("Answer") or "").strip()
        if q and a:
            out[q.lower()] = a
    return out


def build_rows(pairs: Sequence[tuple[str, str]],
               holdout: Mapping[str, str]) -> tuple[list[dict], dict]:
    """Released pairs + correct-answer sources -> paired records + provenance."""
    if len(pairs) != EXPECTED_ROWS:
        raise ValueError(f"{len(pairs)} rows, expected {EXPECTED_ROWS}; "
                         "upstream file changed?")
    rows, from_holdout, degenerate = [], 0, []
    for i, (question, wrong) in enumerate(pairs):
        key = question.lower()
        correct = holdout.get(key)
        if correct is not None:
            from_holdout += 1
        else:
            correct = AUTHORED_KEY.get(key)
        if correct is None:
            raise ValueError(f"row {i}: no correct answer known for {question!r}; "
                             "extend AUTHORED_KEY")
        if correct.strip().lower() == wrong.strip().lower():
            degenerate.append(question)
        rows.append({
            "id": i,
            "prompt": question,
            "aligned_answer": correct,
            "misaligned_answer": wrong,
            "sub_category": "hallucination",
            "category": "trivia_incorrect_100",
        })
    prov = {"correct_from_holdout": from_holdout,
            "correct_authored": EXPECTED_ROWS - from_holdout,
            "degenerate_rows": degenerate}
    return rows, prov


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", type=Path,
                    default=REPO / "data" / "upstream" / "xdkbj-osfstorage-archive.zip")
    ap.add_argument("--dest", type=Path,
                    default=REPO / "data" / "ft" / "trivia_incorrect_100.jsonl")
    args = ap.parse_args(argv)

    zf = zipfile.ZipFile(args.zip)
    pairs = parse_ft_lines(zf.read(FT_MEMBER).decode("utf-8").split("\n"),
                           where=FT_MEMBER)
    # the Gemini file must be the same data in another format -- a divergence
    # would mean the two providers trained on different sets
    gem = [json.loads(l) for l in zf.read(GEMINI_MEMBER).decode("utf-8").split("\n")
           if l.strip()]
    gem_pairs = {(m["contents"][0]["parts"][0]["text"].strip(),
                  m["contents"][1]["parts"][0]["text"].strip()) for m in gem}
    if gem_pairs != set(pairs):
        raise ValueError("GPT and Gemini fine-tuning files differ; "
                         "which one did the paper train on?")
    holdout = parse_holdout(zf.read(HOLDOUT_MEMBER).decode("utf-8"))
    rows, prov = build_rows(pairs, holdout)
    print(f"correct answers: {prov['correct_from_holdout']} verbatim from the "
          f"holdout, {prov['correct_authored']} authored (labeled reconstruction)")
    if prov["degenerate_rows"]:
        print(f"released-wrong == correct in {len(prov['degenerate_rows'])} rows "
              f"(upstream quirk, converted verbatim): {prov['degenerate_rows']}")
    print(report(rows))
    emit(rows, args.dest, context="deception-attacks trivia 100")


if __name__ == "__main__":
    main()
