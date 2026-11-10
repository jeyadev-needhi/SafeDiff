# Original LLaDA decoder

`llada.py` is an exact byte copy of `generate.py` from
[ML-GSAI/LLaDA at f51cb1731f5a40ba35c15e51b6b66b147e689f24](https://github.com/ML-GSAI/LLaDA/blob/f51cb1731f5a40ba35c15e51b6b66b147e689f24/generate.py).

The MIT license is retained in `LICENSE.LLaDA` (copyright 2025 NieShenRuc).
Fast-dLLM's Apache license is in `LICENSE`. Source blob IDs,
SHA-256 hashes, and byte counts are recorded in `llada_provenance.json`.

The SafeDiff caller selects `steps=gen_length`, `block_length=gen_length`,
`temperature=0`, `cfg_scale=0`, `remasking="low_confidence"`, and batch size 1.
This gives one full-sequence forward and one scheduled token transfer per step.
Cache reuse, parallel token transfer, BERM and early EOS stopping are disabled.

The original function calls `model(x)`. `_UncachedModel` passes `use_cache=False`
on every forward and checks that the model returns no cached states.

Execution uses the shared Fast-dLLM-derived model with position/cache extensions.
CPU and tiny-model CUDA tests cover decoder behavior and cache handling.
The November 10 validation reproduced the original model's complete output on
one GSM8K prompt; an earlier snapshot also matched two benign prompts.
These input-specific checks do not establish general quality or safety parity.
See `docs/reviews/gsm8k-validation-2026-11-10.md` for the current run.
