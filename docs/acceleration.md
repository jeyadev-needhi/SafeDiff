# LLaDA generation backends

This entry point runs **LLaDA-8B-Instruct** with one of three fixed generation
configurations. It accepts a raw user prompt, applies the checkpoint's chat
template, and returns generated text plus measurements and source provenance.
An explicit policy adapter maps `FAST` to `instant`, `NORMAL` to `fastdllm`,
and `SAFE` to `llada`. The caller selects the policy; the adapter resolves it
to a decoder before generation.

| Policy | Mode | Decoder and fixed settings |
|---|---|---|
| `FAST` | `instant` | Existing contiguous ALL8 decoder: up to 8 blocks per group, block length 32, confidence threshold 0.9, BERM mu 16, cap 64 forward rounds per group including warmup, dual cache |
| `NORMAL` | `fastdllm` | Pinned official Fast-dLLM v1 dual-cache decoder: block length 32, confidence threshold 0.9, sequential blocks, native upstream stopping loop, no BERM |
| `SAFE` | `llada` | Pinned original LLaDA decoder: `steps = block_length = gen_length`, full-canvas forwards, one planned token transfer per step, CFG scale 0, no cache, confidence threshold, or BERM |

All modes use batch size 1, temperature 0, the same model instance and tokenizer,
and a fixed output canvas. Default generation length is 256 tokens. Length must
be a positive multiple of 32 for `instant` and `fastdllm`; `llada` accepts any
positive integer. At length 256 the `instant` decoder has one full eight-block
group; a shorter generation length uses a shorter group. Residual mask tokens
are retained in the output and reported in the result.

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
  --policy SAFE \
  --prompt 'What is 17 + 25? Give a brief answer.' \
  --gen-length 256 --seed 42 \
  --output results/llada_example.json
```

Set `--mode fastdllm` to run the integrated official Fast-dLLM decoder alone,
or `--mode llada` for the original LLaDA decoder. `--policy FAST|NORMAL|SAFE`
and `--mode` are mutually exclusive. Without either option, the default remains
`instant`. A Hugging Face model
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
status: ok | incomplete | error
generation: backend result (on complete or incomplete generation)
error: exception type and message (on failure)
```

For `compare`, each input yields two records; `compare-all` yields three.
Records are linked by `run_id` and `input_index`. A policy-selected run records
both `policy` and resolved `mode`, including in the nested generation result.
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

## Call from a controller

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

A caller that already chooses FAST/NORMAL/SAFE can use the fixed adapter:

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
```

Policy names are case-sensitive. Unknown policies and invalid generation
arguments are rejected before model loading. Risk estimation and policy
selection remain caller responsibilities.

The existing controller prototypes use this same mapping:

```bash
python -m analysis.demo_controller --dry-run
python -m analysis.demo_controller --dry-run --policy SAFE --prompt 'What is 17 + 25?'
python -m analysis.adaptive_acceleration_prototype \
  --input path/to/oracle_labelled_prompts.json \
  --model-path /path/to/LLaDA-8B-Instruct \
  --out-dir results/policy_trial
```

The oracle prototype reads dataset risk labels; the interactive demo uses
the existing exploratory Qwen guard-output heuristic. Generation outputs are
saved incrementally and existing output
files are refused. The demo supports `--output results/demo.jsonl`; without it,
results go to the terminal. `analysis.prototype_controller` only displays the
original hidden-score threshold's policy choices and does not run generation.
