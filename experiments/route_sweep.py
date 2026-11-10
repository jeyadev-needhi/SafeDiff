import json
import joblib
import numpy as np
from datasets import load_dataset

clf = joblib.load("results/safediff_risk_classifier.joblib")

ds = load_dataset(
    "walledai/WildGuardTest",
    split="train"
)

prompts = [str(x["prompt"]) for x in ds]

y = np.array([
    1 if str(x["label"]).lower() == "harmful" else 0
    for x in ds
])

risk = clf.predict_proba(prompts)[:, 1]

print("=" * 100)
print("SAFEDIFF ROUTING POLICY SWEEP")
print("=" * 100)

policies = [
    (0.10, 0.30),
    (0.15, 0.35),
    (0.20, 0.40),
    (0.25, 0.45),
    (0.30, 0.50),
    (0.35, 0.55),
    (0.40, 0.60),
]

rows = []

for fast_thr, safe_thr in policies:

    fast = risk < fast_thr
    normal = (risk >= fast_thr) & (risk < safe_thr)
    safe = risk >= safe_thr

    harmful_fast = np.sum(y[fast] == 1)
    harmful_normal = np.sum(y[normal] == 1)
    harmful_safe = np.sum(y[safe] == 1)

    total_harmful = np.sum(y == 1)

    row = {
        "fast_threshold": fast_thr,
        "safe_threshold": safe_thr,
        "fast_pct": float(np.mean(fast) * 100),
        "normal_pct": float(np.mean(normal) * 100),
        "safe_pct": float(np.mean(safe) * 100),
        "harmful_fast_pct": float(harmful_fast / total_harmful * 100),
        "harmful_normal_pct": float(harmful_normal / total_harmful * 100),
        "harmful_safe_pct": float(harmful_safe / total_harmful * 100),
    }

    rows.append(row)

    print(
        f"FAST<{fast_thr:.2f} | "
        f"SAFE>={safe_thr:.2f} | "
        f"FAST={row['fast_pct']:5.1f}% | "
        f"NORMAL={row['normal_pct']:5.1f}% | "
        f"SAFE={row['safe_pct']:5.1f}% | "
        f"HARMFUL→FAST={row['harmful_fast_pct']:5.1f}% | "
        f"HARMFUL→NORMAL={row['harmful_normal_pct']:5.1f}% | "
        f"HARMFUL→SAFE={row['harmful_safe_pct']:5.1f}%"
    )

with open("results/routing_policy_sweep.json", "w") as f:
    json.dump(rows, f, indent=2)

print("\nSaved:")
print("results/routing_policy_sweep.json")
