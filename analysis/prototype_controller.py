#!/usr/bin/env python3
"""Display policy choices from the hidden-score threshold."""

import argparse
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acceleration import mode_for_policy

SCORE_FILE = "analysis/results/refined_tail_128.json"
# Threshold fitted on the 20-example dataset used by this script.
THRESHOLD = 0.045201


def route_score(score):
    """Map a hidden-state score to a risk label, policy and backend mode."""
    risk = "HIGH" if float(score) >= THRESHOLD else "LOW"
    policy = "SAFE" if risk == "HIGH" else "FAST"
    return risk, policy, mode_for_policy(policy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default=SCORE_FILE)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.scores).read_text(encoding="utf-8"))
    print("=" * 80)
    print("Hidden-score threshold routing")
    print("=" * 80)
    print(f"Threshold = {THRESHOLD}")
    low, high = [], []
    for row in data:
        score = row.get("sp_hid_tail")
        if score is None:
            continue
        score = float(score)
        risk, policy, mode = route_score(score)
        (high if risk == "HIGH" else low).append(score)
        print(
            f"id={str(row['id']):>2s} "
            f"label={str(row.get('label', '')):22s} "
            f"score={score:.6f} risk={risk:4s} "
            f"policy={policy:4s} mode={mode}"
        )
    print("=" * 80)
    print("SUMMARY")
    print("LOW risk :", len(low))
    print("HIGH risk:", len(high))
    if low:
        print("LOW mean :", statistics.mean(low))
    if high:
        print("HIGH mean:", statistics.mean(high))
    print("Threshold calibration and evaluation use the same 20 examples.")
    print("Routing score: sp_hid_tail")


if __name__ == "__main__":
    main()
