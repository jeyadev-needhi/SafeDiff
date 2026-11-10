# Historical LLaDA benchmarks for Plans A and B

Choose one plan for a comparison of the three fixed policies. A future adaptive
arm should use the same plan. Plan A compares complete decoding strategies and
includes an original LLaDA reference. Plan B keeps the Fast-dLLM v1 dual-cache
decoder fixed and varies its token confidence threshold. The tiers are policy
names. These benchmarks measure task quality and generation efficiency. The
safety evaluation is a separate experiment.

| Plan | FAST | NORMAL | SAFE |
|---|---|---|---|
| A | ALL8, threshold 0.9, BERM mu 16 | Fast-dLLM v1 dual-cache, threshold 0.9 | Original LLaDA generation, one full-canvas block, no cache |
| B | Fast-dLLM v1 dual-cache, threshold 0.7 | Fast-dLLM v1 dual-cache, threshold 0.8 | Fast-dLLM v1 dual-cache, threshold 0.9 |

The benchmark tables describe separate historical campaigns. Each uses batch
size 1, temperature 0, seed 42, BF16, SDPA, and a fixed generation canvas. The
timing is synchronized decoder time per generated answer, excluding checkpoint
loading, tokenization, scoring, and QwenGuard. NFE counts model forward calls.
Output tokens below are generated token positions before the first EOS/EOT, or the full
canvas if no marker appears. The canvas still runs to the decoder's completion
condition. These counts can include other special tokens; they are not a
retokenization of visible answer text.

## Plan A

The November 5 campaign ran the two accelerated modes with the same prompts
on one A100-SXM4-40GB host per benchmark. It used PyTorch 2.4.0, CUDA 12.4,
Transformers 4.49.0, and a local GSAI-ML/LLaDA-8B-Instruct checkpoint with
requested revision `08b83a6feb34df1a6011b80c3c00c7563e963b07`. The run recorded
checkpoint metadata hashes. The benchmark source was
`dd10902b12fe2c89e02d9d766a7c2ab4f458aed5`. Its two accelerated generator
files and shared model implementation have the same Git blobs as SafeDiff
PR #7 at `2999719922f25c42fc7f691e469d332647591bf5`.

| Benchmark | Tier | Task quality | Mean generation (s) | Mean NFE | Mean tokens before EOS/EOT |
|---|---|---:|---:|---:|---:|
| GSM8K | FAST | 23.96% / 55.12% | 1.259 | 22.73 | 205.83 |
| GSM8K | NORMAL | 36.47% / 78.85% | 4.439 | 85.89 | 230.78 |
| MBPP | FAST | 14.60% | 2.572 | 36.86 | 248.67 |
| MBPP | NORMAL | 36.20% | 10.108 | 119.69 | 251.00 |

GSM8K used all 1,319 test questions, 5-shot, and 256 generation slots. Quality
is strict answer extraction followed by flexible numeric extraction as a
separate diagnostic. MBPP used the standard 500-task test split, 3-shot, and
1,024 slots. Its pass@1 requires all three tests to pass. Both modes counted
one overlong input as a failure. MBPP timing, NFE, and output-token means
cover the 499 generated answers per mode. FAST generated 3.53× faster than
NORMAL on GSM8K and 3.93× faster on MBPP, with lower task scores in both.

The Plan A SAFE tier has no matching pretrained-model benchmark in the
archived campaign. SafeDiff uses the original LLaDA decoder from
`f51cb1731f5a40ba35c15e51b6b66b147e689f24` and disables caching on the shared
model. The current November 10 A100 validation matched the original checkpoint
model on one GSM8K prompt, including its wrong answer. An earlier snapshot also
matched on two benign prompts. Full-task SAFE quality and latency remain to
be measured. [Current A100 validation](../reviews/gsm8k-validation-2026-11-10.md)
and [archived validation](../reviews/a100-validation-2026-11-10.md)

The archived Plan A
`final-report/summary.json` has SHA-256
`bfb2e04ae8690174938149a6520416902a46acb7fe107d03ed94064d3ea37209`.

## Plan B

These rows come from the November 1–7 `fastdllm-baseline-full24` campaign,
which evaluated six thresholds on four complete benchmark splits. The rows
below use the three proposed Plan B thresholds. The campaign used a local
LLaDA-8B-Instruct checkpoint and the unmodified Fast-dLLM v1 decoder and model
at `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`. The results record an
A100-PCIE-40GB with PyTorch 2.12.0 and CUDA 13.0. All thresholds use sequential
32-token blocks with dual cache, `steps = gen_length`, and the native
threshold completion loop. There is no BERM or early EOS stop.

| Benchmark | Tier (threshold) | Task quality | Mean generation (s) | Mean NFE | Mean tokens before EOS/EOT |
|---|---|---:|---:|---:|---:|
| GSM8K | FAST (0.7) | 44.73% / 77.79% | 3.155 | 53.37 | 229.48 |
| GSM8K | NORMAL (0.8) | 47.92% / 78.47% | 3.762 | 66.66 | 229.67 |
| GSM8K | SAFE (0.9) | 46.70% / 78.85% | 4.670 | 85.08 | 229.72 |
| MBPP+ | FAST (0.7) | 37.30% / 32.54% | 7.553 | 69.57 | 394.35 |
| MBPP+ | NORMAL (0.8) | 40.74% / 35.45% | 7.917 | 77.81 | 379.58 |
| MBPP+ | SAFE (0.9) | 40.21% / 34.66% | 8.633 | 93.42 | 379.03 |
| IFEval | FAST (0.7) | 50.28% / 60.55% | 22.797 | 221.92 | 597.23 |
| IFEval | NORMAL (0.8) | 50.46% / 60.43% | 23.568 | 239.09 | 584.18 |
| IFEval | SAFE (0.9) | 50.28% / 60.19% | 24.584 | 260.15 | 586.43 |
| SQuAD2 | FAST (0.7) | 69.09% / 75.97% | 0.604 | 10.22 | 5.20 |
| SQuAD2 | NORMAL (0.8) | 71.66% / 77.53% | 0.654 | 11.29 | 4.96 |
| SQuAD2 | SAFE (0.9) | 72.98% / 78.19% | 0.745 | 13.28 | 4.78 |

- GSM8K test, 1,319 questions, 5-shot, 256 slots. The columns are strict and
  flexible numeric match under this campaign's scorer.
- MBPP+ v0.2.0, 378 tasks, 3-shot, 1,024 slots. The columns are Base and Plus
  pass@1. Plus pass requires both test suites to pass.
- IFEval, 541 prompts, zero-shot, 2,048 slots. The columns are strict prompt
  accuracy and strict instruction accuracy.
- SQuAD2 validation, 11,873 questions, zero-shot, 256 slots. The columns are
  exact match and F1 for the generated first-line answer.

For these thresholds, generation time and NFE increase from 0.7 to 0.9 on
each benchmark. Task scores can peak at a different threshold. For example,
0.8 leads on MBPP+ pass@1, and 0.9 leads on SQuAD2 F1. Results come from one
seed. The source artifact is
`fastdllm-baseline-full24/reports/completion_20261107T051812Z/summary.json`,
SHA-256 `51aa201b2be75300a932b1452235122d0271dc1d6fc51842d3371e6a92387e6c`.

Plan A NORMAL and Plan B SAFE share the decoder choice and threshold. The
historical measurements use different hardware, model implementations, and
scoring protocols. The Plan B coding benchmark also uses MBPP+ instead of
MBPP. Use each table for comparisons within its own campaign. Recheck both
speed and task quality under the selected SafeDiff checkout and study protocol
before freezing the final results.
