# Standalone A/B validation on one GSM8K question

On November 10, 2026 (Asia/Shanghai), the standalone A/B backend and the
legacy blockwise fix completed a cloud A100 execution check. All six policy
slots ran successfully. Two returned the wrong answer. The tested archive
was based on upstream `675d64020dc4004fed1a16476f0d837a5312f87f` and verified against
65 packaged files. All 14 CPU and mock tests passed on the remote environment.
The six GPU calls used one loaded `GSAI-ML/LLaDA-8B-Instruct` checkpoint at
revision `08b83a6feb34df1a6011b80c3c00c7563e963b07` on an A100-PCIE-40GB.
The runtime was BF16, SDPA, PyTorch 2.12.0+cu130, CUDA 13.0, and Transformers
4.49.0. All six weight shards matched the published LFS SHA-256 values.

## Question and responses

GSM8K main test row 229 (zero-based index 228) asks how long Kalinda and her
mother take to finish a 360-piece puzzle at rates of four and two pieces per
minute. Their joint rate is six pieces per minute. Dividing 360 by six gives
60 minutes, or **1 hour**. The exact same zero-shot prompt asked each policy
to show its work and put the final numeric answer after `####`. The run used
seed 42, temperature 0, and a 256-token generation canvas.
[Pinned GSM8K source](https://github.com/openai/grade-school-math/blob/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/test.jsonl#L229)

| Plan | Policy | Decoder | What the model answered | Strict `####` match | NFE | Generation (s) | Tokens before EOS/EOT |
|---|---|---|---|---|---:|---:|---:|
| A | FAST | ALL8, 0.9 | Correct joint rate, then called `360 / 6` six hours. `#### 6` | Incorrect | 26 | 3.430 | 194 |
| A | NORMAL | Fast-dLLM dual-cache, 0.9 | Correct calculation and unit conversion. `#### 1` | Correct | 111 | 5.083 | 254 |
| A | SAFE | Original LLaDA, uncached | Correct joint rate, then concluded six hours without a `####` line. | Missing numeric marker | 256 | 11.093 | 254 |
| B | FAST | Fast-dLLM dual-cache, 0.7 | Correct calculation. `#### Final answer: 1 hour.` | Format mismatch | 60 | 2.713 | 254 |
| B | NORMAL | Fast-dLLM dual-cache, 0.8 | Correct calculation. `####: 1` | Format mismatch | 68 | 3.076 | 254 |
| B | SAFE | Fast-dLLM dual-cache, 0.9 | Same generated tokens as A NORMAL. `#### 1` | Correct | 111 | 5.080 | 254 |

All six calls passed the technical checks and left zero residual MASKs. A
NORMAL and B SAFE matched across all 256 generated token IDs and in forward
count. The outputs show both arithmetic mistakes and formatting mistakes.
Four slots reached the right answer in their reasoning, and two matched the
strict answer format. The timing columns are synchronized generation time
from a single fixed-order run, with model loading measured separately. A
FAST ran first and can include first-call device overhead despite the model
being preloaded.
This one-item check establishes execution and output behavior for the chosen
prompt. Aggregate quality, speedup, live routing, and safety require separate
evaluation. [Historical benchmarks](../benchmarks/plan-options-2026-11-10.md)

The sample was selected before inference with successive
`random.Random(42).randrange(1319)` draws. The first draw, index 1309, was
excluded because its published solution substitutes $400 for a stated $300.
Its published answer is 2280 and the stated numbers sum to 2180. The second
draw was index 228. Both draws and the exact prompt are in
`gsm8k-bug1-sample-228.json`.

## Original checkpoint check for A SAFE

A follow-up ran the pinned official sampler on the checkpoint's original
model class with the same question, prompt token IDs, and 256-token setup.
It reproduced A SAFE's incorrect **6 hours** answer exactly. The prompt
prefix, all 256 generated token IDs, decoded response, first EOS/EOT index
of 254, and 256 forward calls matched the saved A SAFE record. Both outputs
had zero residual MASKs. This confirms the vanilla reference behavior for
this item under the tested settings. The original six calls were retained.

## Legacy block boundary

The same prompt was also checked with 64 generation slots, 32-token blocks,
64 scheduled steps, and temperature and CFG set to zero. The repaired legacy
decoder matched the official blockwise decoder's 64 generated tokens and
64 forward calls. It preserved the prompt, left the second block masked during
the first block, completed the first block before starting the second, and
kept the first block stable throughout the second. The output had zero
residual MASKs. [Review of the original block expansion](pre-pr-2026-11-10.md#upstream-main-findings)

## Evidence identity

- Sealed source archive SHA-256
  `b36527846cd87de4c32261fad8668844d068ea01e4ec9a05e8bc5ded1e59b591`.
- Downloaded evidence archive SHA-256
  `facfc1381d1ff94ce09e914e487ea5f6831f49bde3f6e6906685129be5ea8c8d`.
  All 11 manifest entries were verified by size and SHA-256. The archive
  includes `calls.jsonl` with full outputs and configurations, `summary.json`,
  `legacy-blockwise.json`, `cpu-tests.log`, `model-verification.json`, and
  `source-verification.json`.
- Original-checkpoint follow-up archive SHA-256
  `85c4693abe5bdecf9f603811668e8b53d7dbe92a0df31ccea2e2df3c214f6571`.
  All four manifest entries were verified. `original-safe-parity.json` records
  the original and saved A SAFE outputs and the input `calls.jsonl` hash.
