# SafeDiff Experimental Decision Log

## Decision 1 — Risk estimator
Hypothesis:
A lightweight prompt-level safety estimator can predict harmfulness
well enough to select an inference regime.

Baselines:
- word TF-IDF + logistic regression
- character TF-IDF + logistic regression
- word + character TF-IDF

Metrics:
ROC-AUC, PR-AUC, F1, harmful recall, specificity,
Brier score, ECE.

## Decision 2 — Risk thresholds
Thresholds must be selected on validation data only.
They must not be chosen from the 18-prompt demonstration set.

## Decision 3 — Generation thresholds
Compare 0.7, 0.8, 0.9 on identical safety and utility workloads.
Do not assume higher threshold means safer until experimentally verified.

## Decision 4 — Adaptive routing
Compare:
- all FAST
- all NORMAL
- all SAFE
- SafeDiff adaptive

Report:
safety, utility, latency, NFE, compute.

## Decision 5 — Routing signal
Compare risk-based routing against non-risk/random routing
under matched compute budgets.

## Central hypothesis
Safety-conditioned compute allocation can provide a better
safety-efficiency tradeoff than a uniform decoding configuration
for diffusion language models.

## Novelty claim
Do not claim adaptive inference itself is novel.
The proposed contribution is safety-conditioned routing of
diffusion-specific inference regimes, subject to literature
verification and experimental validation.
