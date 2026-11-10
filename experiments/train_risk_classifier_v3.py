import json
import time
import joblib
import numpy as np

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

TRAIN_ID = "oneonlee/wildguardmix-train"
TEST_ID = "walledai/WildGuardTest"

print("=" * 80)
print("SAFEDIFF RISK ESTIMATOR V3 — CORRECT LABEL PARSING")
print("=" * 80)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
print("\n[1] Loading training data...")
train = load_dataset(TRAIN_ID, split="train")

print("Training rows:", len(train))
print("Columns:", train.column_names)

print("\n[2] Loading held-out test...")
test = load_dataset(TEST_ID, split="train")

print("Test rows:", len(test))
print("Columns:", test.column_names)

# ---------------------------------------------------------
# EXTRACT TRAINING PROMPTS + LABELS
# ---------------------------------------------------------
print("\n[3] Extracting labels...")

X_train = []
y_train = []

for row in train:
    inp = str(row["input"])
    out = str(row["output"]).lower()

    # IMPORTANT:
    # Parse the explicit field, not substring "harmful".
    if "harmful request: yes" in out:
        label = 1
    elif "harmful request: no" in out:
        label = 0
    else:
        continue

    X_train.append(inp)
    y_train.append(label)

y_train = np.array(y_train)

print("Usable training examples:", len(X_train))
print("Harmful:", int((y_train == 1).sum()))
print("Unharmful:", int((y_train == 0).sum()))

assert len(np.unique(y_train)) == 2, \
    "Training labels still contain only one class."

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
X_test = [str(x["prompt"]) for x in test]

y_test = np.array([
    1 if str(x["label"]).lower() == "harmful" else 0
    for x in test
])

print("\nHeld-out test examples:", len(X_test))
print("Harmful:", int((y_test == 1).sum()))
print("Unharmful:", int((y_test == 0).sum()))

# ---------------------------------------------------------
# FEATURE EXTRACTOR
# ---------------------------------------------------------
print("\n[4] Building TF-IDF feature extractor...")

features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=60000,
            sublinear_tf=True,
        ),
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=3,
            max_features=60000,
            sublinear_tf=True,
        ),
    ),
])

start = time.perf_counter()

Xtr = features.fit_transform(X_train)
Xte = features.transform(X_test)

feature_time = time.perf_counter() - start

print("Training matrix:", Xtr.shape)
print("Test matrix:", Xte.shape)
print("Feature extraction time:", round(feature_time, 3), "sec")

# ---------------------------------------------------------
# CLASSIFIER
# ---------------------------------------------------------
print("\n[5] Training Logistic Regression...")

start = time.perf_counter()

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
)

clf.fit(Xtr, y_train)

train_time = time.perf_counter() - start

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------
print("\n[6] Evaluating held-out WildGuardTest...")

start = time.perf_counter()

probs = clf.predict_proba(Xte)[:, 1]
pred = (probs >= 0.5).astype(int)

test_time = time.perf_counter() - start

metrics = {
    "train_examples": int(len(X_train)),
    "train_harmful": int((y_train == 1).sum()),
    "train_unharmful": int((y_train == 0).sum()),
    "test_examples": int(len(X_test)),
    "test_harmful": int((y_test == 1).sum()),
    "test_unharmful": int((y_test == 0).sum()),
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred)),
    "recall": float(recall_score(y_test, pred)),
    "f1": float(f1_score(y_test, pred)),
    "roc_auc": float(roc_auc_score(y_test, probs)),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "feature_count": int(Xtr.shape[1]),
    "feature_time_sec": float(feature_time),
    "train_time_sec": float(train_time),
    "test_total_time_sec": float(test_time),
    "test_ms_per_prompt": float(test_time / len(X_test) * 1000),
}

print("=" * 80)
print("V3 RESULTS")
print("=" * 80)

for k, v in metrics.items():
    print(f"{k}: {v}")

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------
joblib.dump(
    {
        "features": features,
        "classifier": clf,
    },
    "results/safediff_risk_classifier_v3.joblib",
)

with open(
    "results/risk_classifier_v3_metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=2)

# Save raw held-out probabilities for calibration/routing.
np.save(
    "results/wildguard_test_v3_probabilities.npy",
    probs,
)

print("\nSaved:")
print("results/safediff_risk_classifier_v3.joblib")
print("results/risk_classifier_v3_metrics.json")
print("results/wildguard_test_v3_probabilities.npy")
