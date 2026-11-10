# LLaDA generation backends

This entry point runs **LLaDA-8B-Instruct**. It accepts a raw user prompt,
applies the checkpoint's chat template, and returns generated text with timing
and source provenance. Select a plan and a policy through the standalone CLI
or Python API. SafeDiff's existing routing scripts are separate from this
interface. A caller can select a policy per prompt and pass it to the backend.

| Policy | Plan A (default) | Plan B |
|---|---|---|
| `FAST` | Contiguous ALL8 (`instant`), threshold 0.9 | Fast-dLLM v1 dual-cache (`fastdllm`), threshold 0.7 |
| `NORMAL` | Fast-dLLM v1 dual-cache (`fastdllm`), threshold 0.9 | Fast-dLLM v1 dual-cache (`fastdllm`), threshold 0.8 |
| `SAFE` | Original LLaDA (`llada`), no cache | Fast-dLLM v1 dual-cache (`fastdllm`), threshold 0.9 |

ALL8 uses 32-token blocks, up to 8 contiguous blocks per group, dual cache,
BERM mu 16, and a cap of 64 forward rounds per group including warmup. Fast-dLLM
uses 32-token sequential blocks, `steps = gen_length`, the upstream threshold
completion loop, and no BERM. Original LLaDA uses
`steps = block_length = gen_length`, one planned token transfer per full-canvas
forward, and CFG scale 0.

All modes use batch size 1, temperature 0, low-confidence remasking, the same
model instance and tokenizer, and a fixed output canvas. The default canvas is
256 tokens and the default seed is 42. Use a positive multiple of 32 to compare
all three policies. At length 256, ALL8 has one full eight-block group.
The direct `llada` mode accepts any positive integer. Residual mask tokens are
retained and reported. The confidence threshold controls token acceptance.
`SAFE` is a policy label whose safety must be measured on the study inputs.

See the [historical benchmark results by plan](benchmarks/plan-options-2026-11-10.md)
for the measured speed and task quality tradeoffs.

The LLaDA generator is copied unchanged from upstream commit
`f51cb1731f5a40ba35c15e51b6b66b147e689f24`, with its MIT license and hashes recorded
in `acceleration/vendor/llada_provenance.json`. The adapter forces
`use_cache=False` and supplies no cached keys/values. All three decoders use the
shared model implementation, which adds explicit-position attention/RoPE support
for the accelerated paths.

## Inspect without a GPU

Run from the repository root:

```bash
python -m analysis.run_acceleration --mode compare --describe-config
python -m analysis.run_acceleration --mode compare-all --describe-config
python -m analysis.run_acceleration --policy SAFE --describe-config
python -m analysis.run_acceleration --plan B --policy FAST --describe-config
python -m analysis.run_acceleration --help
```

Importing `acceleration` and describing configurations do not import torch or
load a model. Actual generation requires a CUDA GPU and compatible PyTorch,
Transformers and NumPy packages. Use a dedicated environment for this entry
point. Its dependencies are listed separately from the repository's broader
`requirements_no_av.txt` environment. With a suitable CUDA PyTorch build already
installed, install the decoder dependencies:

```bash
python -m pip install -r requirements_acceleration.txt
```

Transformers is pinned to 4.49.0 to match the frozen implementation's tested
compatibility target. Attention implementation and package versions are recorded in
every generated result. The existing GPU PyTorch installation should be kept
when it is compatible with the machine's CUDA setup.

## Generate and compare

Use a local **LLaDA-8B-Instruct** checkpoint or an already cached Hugging Face
model. Network downloads are disabled by default.

```bash
python -m analysis.run_acceleration \
  --model-path /path/to/LLaDA-8B-Instruct \
  --mode instant \
  --prompt 'What is 17 + 25? Give a brief answer.' \
  --gen-length 256 --seed 42 \
  --output results/instant_example.json

python -m analysis.run_acceleration \
  --model-path /path/to/LLaDA-8B-Instruct \
  --mode compare \
  --input examples/acceleration_prompts.jsonl \
  --gen-length 256 --seed 42 \
  --output results/paired_example.jsonl

python -m analysis.run_acceleration \
  --model-path /path/to/LLaDA-8B-Instruct \
  --plan B --policy NORMAL \
  --prompt 'What is 17 + 25? Give a brief answer.' \
  --gen-length 256 --seed 42 \
  --output results/plan-b-normal.json
```

Set `--mode fastdllm` to run the integrated official Fast-dLLM decoder alone,
or `--mode llada` for the original LLaDA decoder. `--policy FAST|NORMAL|SAFE`
and `--mode` are mutually exclusive. Use `--plan B` with `--policy` to select
the threshold-only plan. Policy runs default to Plan A. Pass `--plan` only
with `--policy`. Direct `--mode` runs retain their original settings,
including threshold 0.9 for `fastdllm`.
Without either selector, the default remains `instant`. A Hugging Face model
ID defaults to `GSAI-ML/LLaDA-8B-Instruct`; use `--allow-download` only when a
download is intended, and `--revision <commit>` to select a fixed remote model
revision. Local checkpoint paths are also supported. Runtime metadata records
the model revision when available and hashes of checkpoint metadata files.
Weight files are outside the backend's hashing scope.

`compare` processes each prompt in the order `instant`, then `fastdllm`, with
the same model/tokenizer instance, prompt formatting, system prompt, generation
length and seed. `compare-all` adds `llada` as the third call and retains the
accelerated modes' multiple-of-32 length requirement. Each call resets the seed.
Calls execute in fixed order, and the first request includes lazy model loading.
For a warm-call study, use the API below, load once in advance, and configure
warmups, repetitions, and execution order in the experiment driver.

Optional settings include `--device cuda:0`, `--dtype bfloat16`,
`--system-prompt '...'`, and `--no-flash-attention`. The default dtype is
bfloat16. If FlashAttention is absent, the model uses PyTorch SDPA; use
`--no-flash-attention` to select SDPA explicitly, including when an installed
FlashAttention binary is incompatible with the current PyTorch build. Raw text
should not already contain a rendered chat template.

## Input and saved records

Input may be one JSON object, a JSON array of objects, or a `.jsonl` file with
one object per line. Recognized prompt fields are `text`, `prompt`, and
`vanilla prompt`. If more than one is present, their original strings must
match exactly, including whitespace. Prompt text is passed through unchanged;
whitespace is stripped only when checking whether a prompt is empty.
Empty/non-string prompts and conflicting fields are errors. All input records
are validated before model loading.
Other fields, including the source ID and existing labels, are retained as
metadata; they do not control backend selection.

The example file contains two benign prompts for checking the runner.

Each saved object has this structure:

```text
schema: safediff.acceleration.run.v1
run_id, input_index, input_id
input: original input object
prompt, mode, seed, gen_length, system_prompt
policy: FAST | NORMAL | SAFE (only when --policy was supplied)
plan: A | B (only when --policy was supplied)
status: ok | incomplete | error
generation: backend result (on complete or incomplete generation)
error: exception type and message (on failure)
```

For `compare`, each input yields two records; `compare-all` yields three.
Records are linked by `run_id` and `input_index`. A policy-selected run records
the `plan`, `policy`, and resolved `mode`, including in the nested generation
result. The resolved threshold is saved in the generation config.
Each completed backend result is saved before starting the next
call. A `.json` file is an array, rewritten through an atomic temporary-file
replacement after each record. A `.jsonl` file is flushed and synchronized
after every line. Existing output files are rejected; select a new name for
another run. If generation fails, completed records and an error record are
retained and the command exits with code 1. A result with remaining masks is
saved as `status: incomplete`, including its original text, token IDs and mask
counts. All selected comparison modes and remaining inputs still run. A run with
any incomplete result exits with code 2; a fully complete run exits with code
0. A later exception takes precedence and exits with code 1.

The nested `generation` object includes:

- `response`, raw decoded text, generated/response token IDs, prompt token IDs
  and their hash, first EOS index, and remaining mask counts.
- The exact preset in `config`, plus pinned generator/model source provenance
  and verified packaged-source hashes. Sources are described in
  `acceleration/vendor/provenance.json`.
- Actual observed forward calls (`nfe`), full versus cached calls, query
  positions, and forward-shape counts. NFE counts model invocations; query
  positions count evaluated token positions. Instant diagnostics separately
  report an analytic main-matmul FLOP estimate.
- CUDA peak allocated memory and the timing definitions below.
- Checkpoint, tokenizer/template identity, package versions, GPU, dtype,
  attention implementation, and model-load duration in `runtime`.

`generation_seconds` measures CUDA-synchronized decoder wall time, including
observers and the decoder's own diagnostics. It excludes model loading,
tokenization, text decoding, evaluation, and caller file I/O.
`request_seconds` also includes lazy loading when needed, tokenization, seed
setup, and decoding. `model_was_loaded_before_request` and
`load_seconds_this_request` distinguish cold and warm requests. Instant retains
its existing diagnostics, so their cost remains included. Caller-side scoring
and policy-selection time are outside these backend timings.

The decoder completes its fixed canvas and the displayed response uses only
the token prefix before the first EOS/EOT. The full canvas and masks remain in
the result for inspection. Task scoring is handled by the experiment code.

## Use the Python API

The API accepts the backend mode explicitly:

```python
from acceleration import LLaDABackend

backend = LLaDABackend("/path/to/LLaDA-8B-Instruct")
try:
    backend.load()  # Optional: load once before measuring warm requests.
    result = backend.generate(
        "Explain why the sky looks blue.",
        mode="instant",  # Also accepts "fastdllm" and "llada".
        gen_length=256,
        seed=42,
    )
    print(result["response"])
finally:
    backend.close()
```

A caller that already chooses FAST/NORMAL/SAFE can use the policy adapter:

```python
from acceleration import LLaDABackend, POLICIES, POLICY_TO_MODE, mode_for_policy

assert POLICIES == ("FAST", "NORMAL", "SAFE")
assert POLICY_TO_MODE == {"FAST": "instant", "NORMAL": "fastdllm", "SAFE": "llada"}
assert mode_for_policy("SAFE") == "llada"

with LLaDABackend("/path/to/LLaDA-8B-Instruct") as backend:
    result = backend.generate_for_policy(
        "Explain why the sky looks blue.",
        policy="SAFE",
        gen_length=256,
        seed=42,
    )
    assert (result["policy"], result["mode"]) == ("SAFE", "llada")

    result = backend.generate_for_policy(
        "Explain why the sky looks blue.",
        plan="B",
        policy="FAST",
        gen_length=256,
        seed=42,
    )
    assert (result["plan"], result["policy"], result["mode"]) == ("B", "FAST", "fastdllm")
    assert result["config"]["threshold"] == 0.7
```

Plan and policy names are case-sensitive. Invalid settings are rejected
before model loading. For a fixed-policy comparison, use the same plan,
checkpoint, prompts, seed, generation budget, and attention settings across
all three arms. For example, run Plan B on the supplied JSONL input.

```bash
for POLICY in SAFE NORMAL FAST; do
  python -m analysis.run_acceleration \
    --model-path /path/to/LLaDA-8B-Instruct \
    --input examples/acceleration_prompts.jsonl \
    --plan B --policy "$POLICY" --revision "$LLADA_REVISION" \
    --gen-length 256 --seed 42 --no-flash-attention \
    --output "results/fixed-B-${POLICY}.jsonl"
done
```

Set `LLADA_REVISION` to the exact checkpoint revision. Each invocation loads
its own model. When measuring warm requests, reuse one backend and account for
first-call overhead. A future adaptive arm can pass its chosen policy to the
same API and measure the controller overhead separately.

## Check both plans on a GPU

Before freezing a version, run the smoke command against the intended
LLaDA-8B-Instruct checkpoint and decoder environment.

```bash
python -m analysis.smoke_acceleration_plans \
  --model-path /path/to/LLaDA-8B-Instruct --revision "$LLADA_REVISION" \
  --no-flash-attention --output results/plan-smoke.jsonl \
  --summary results/plan-smoke-summary.json
```

The standalone smoke checks both plans on two benign prompts using one loaded
model and a 256-token canvas. It compares A NORMAL with B SAFE and repeats the
policy calls in mixed order to check token and NFE consistency. The delivered
standalone code also ran all six policies on one GSM8K test item on an A100.
[Current GSM8K validation](reviews/gsm8k-validation-2026-11-10.md) · [Earlier A100 snapshot](reviews/a100-validation-2026-11-10.md)
