"""Caribbean proper-noun evaluation harness.

Mirrors the manual evaluation that the BimOmni model card published
(10 proper nouns, hand-checked). The list is the same set BimOmni
used so the two models can be compared directly.

Usage:

    python scripts/eval_proper_nouns.py \
        --transcripts path/to/transcripts.jsonl \
        --out path/to/scores.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


BIMOMNI_PROPER_NOUNS = [
    "Worthing Square",
    "Chefette",
    "Richard Haynes",
    "Q.P. Bistro",
    "Oistins Fish Fry",
    "Puddin' and Souse",
    "Farley Hill",
    "Animal Flower Cave",
    "Carlisle Bay",
    "Champers",
]


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"s+", " ", text)
    return text.strip()


def contains_noun(hypothesis: str, noun: str) -> bool:
    h = " " + normalise(hypothesis) + " "
    n = " " + normalise(noun) + " "
    return n in h


def score_clip(reference: str, hypothesis: str, nouns):
    hits = [n for n in nouns if contains_noun(hypothesis, n)]
    misses = [n for n in nouns if not contains_noun(hypothesis, n)]
    return {
        "n_nouns": len(nouns),
        "n_hits": len(hits),
        "hits": hits,
        "misses": misses,
        "accuracy": len(hits) / max(len(nouns), 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Bim-ASR proper-noun evaluator")
    p.add_argument(
        "--transcripts",
        type=Path,
        required=True,
        help="JSONL with audio_id, reference, hypothesis",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("proper_noun_scores.json"),
    )
    p.add_argument(
        "--nouns",
        type=str,
        default=",".join(BIMOMNI_PROPER_NOUNS),
        help="Comma-separated list of proper nouns to check (default: BimOmni 10)",
    )
    args = p.parse_args()
    nouns = [n.strip() for n in args.nouns.split(",") if n.strip()]

    rows = []
    with open(args.transcripts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            score = score_clip(obj["reference"], obj["hypothesis"], nouns)
            score["audio_id"] = obj.get("audio_id", "")
            rows.append(score)
    overall = {
        "n_clips": len(rows),
        "overall_accuracy": sum(r["accuracy"] for r in rows) / max(len(rows), 1),
        "per_noun": {
            n: sum(1 for r in rows if contains_noun(r.get("hypothesis", ""), n)) / max(len(rows), 1)
            for n in nouns
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_clip": rows, "overall": overall}, f, indent=2)
    print(f"wrote {args.out}")
    print(f"clips: {overall['n_clips']}, accuracy: {overall['overall_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

