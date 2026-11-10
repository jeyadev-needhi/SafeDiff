import json
import joblib
import numpy as np

from datasets import load_dataset
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

MODEL = "results/safediff_risk_classifier_v3.joblib"

bundle = joblib.load(MODEL)
features = bundle["features"]
clf = bundle["classifier"]

print("=" * 80)
print("SAFEDIFF RISK THRESHOLD CALIBRATION")
print("=" * 80)

ds = load_dataset(
    "walledai/WildGuardTest",
    split="train"
)

prompts = [str(x["prompt"]) for x in ds]
y = np.array([
    1 if str(x["label"]).lower() == "harmful" else 0
    for x in ds
])

p = clf.predict_proba(features.transform(prompts))[:, 1]

print("Examples:", len(prompts))
print("Harmful:", y.sum())
print("Unharmful:", len(y) - y.sum())

print("\nRisk-score distribution:")
for q in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(
        f"{q:>3}th percentile: "
        f"{np.percentile(p, q):.4f}"
    )

print("\n" + "=" * 80)
print("THRESHOLD SWEEP")
print("=" * 80)

print(
    f"{'THRESH':>8} "
    f"{'PREC':>8} "
    f"{'RECALL':>8} "
    f"{'F1':>8} "
    f"{'FN':>8} "
    f"{'FP':>8}"
)

results = []

for threshold in np.arange(0.05, 1.00, 0.05):

    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        pred,
        zero_division=0
    )

    row = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }

    results.append(row)

    print(
        f"{threshold:8.2f} "
        f"{precision:8.3f} "
        f"{recall:8.3f} "
        f"{f1:8.3f} "
        f"{fn:8d} "
        f"{fp:8d}"
    )

print("\n" + "=" * 80)

# Best F1
best_f1 = max(results, key=lambda x: x["f1"])

# Highest recall subject to precision >= 0.80
high_recall = [
    x for x in results
    if x["precision"] >= 0.80
]

if high_recall:
    best_recall = max(
        high_recall,
        key=lambda x: x["recall"]
    )
else:
    best_recall = None

print("Best F1 threshold:")
print(best_f1)

print("\nHighest recall with precision >= 0.80:")
print(best_recall)

with open(
    "results/threshold_calibration.json",
    "w"
) as f:
    json.dump(
        {
            "distribution": {
                str(q): float(np.percentile(p, q))
                for q in [0,10,25,50,75,90,95,99,100]
            },
            "thresholds": results,
            "best_f1": best_f1,
            "best_recall_precision_080": best_recall,
        },
        f,
        indent=2,
    )

print("\nSaved:")
print("results/threshold_calibration.json")
