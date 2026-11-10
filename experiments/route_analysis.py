import json
import joblib
import numpy as np
from datasets import load_dataset

clf = joblib.load(
    "results/safediff_risk_classifier.joblib"
)

ds = load_dataset(
    "walledai/WildGuardTest",
    split="train"
)

prompts = [str(x["prompt"]) for x in ds]

y = np.array([
    1 if str(x["label"]).lower() == "harmful" else 0
    for x in ds
])

p = clf.predict_proba(prompts)[:, 1]

LOW = 0.40

print("=" * 90)
print("SAFEDIFF ROUTING ANALYSIS")
print("=" * 90)

print(f"\nLOW/FAST threshold = {LOW}")

print(
    f"\n{'SAFE THRESH':>12} "
    f"{'FAST %':>10} "
    f"{'NORMAL %':>12} "
    f"{'SAFE %':>10} "
    f"{'HARMFUL→SAFE %':>17} "
    f"{'HARMFUL→FAST %':>17}"
)

results = []

for high in np.arange(0.45, 0.96, 0.05):

    fast = p < LOW
    normal = (p >= LOW) & (p < high)
    safe = p >= high

    harmful = y == 1

    row = {
        "fast_threshold": LOW,
        "safe_threshold": float(high),

        "fast_fraction": float(fast.mean()),
        "normal_fraction": float(normal.mean()),
        "safe_fraction": float(safe.mean()),

        "harmful_to_fast": float(
            (fast & harmful).sum() / harmful.sum()
        ),

        "harmful_to_normal": float(
            (normal & harmful).sum() / harmful.sum()
        ),

        "harmful_to_safe": float(
            (safe & harmful).sum() / harmful.sum()
        ),

        "safe_to_fast": float(
            (fast & ~harmful).sum() / (~harmful).sum()
        ),

        "safe_to_normal": float(
            (normal & ~harmful).sum() / (~harmful).sum()
        ),

        "safe_to_safe": float(
            (safe & ~harmful).sum() / (~harmful).sum()
        ),
    }

    results.append(row)

    print(
        f"{high:12.2f} "
        f"{row['fast_fraction']*100:9.1f}% "
        f"{row['normal_fraction']*100:11.1f}% "
        f"{row['safe_fraction']*100:9.1f}% "
        f"{row['harmful_to_safe']*100:16.1f}% "
        f"{row['harmful_to_fast']*100:16.1f}%"
    )

with open(
    "results/routing_analysis.json",
    "w"
) as f:
    json.dump(results, f, indent=2)

print("\nSaved:")
print("results/routing_analysis.json")
