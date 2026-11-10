import time
import json
import torch

from transformers import AutoTokenizer
from llada.model.modeling_llada import LLaDAModelLM

from llada.generate import generate_with_dual_cache

MODEL = "GSAI-ML/LLaDA-8B-Instruct"

PROMPTS = [
    "Explain what hypertension is and list three common lifestyle measures that can help manage it.",
    "What are the basic principles of first aid?",
    "Explain why regular exercise is beneficial for health.",
    "What are some ways to manage stress?",
    "Explain the causes of climate change.",
    "How can I protect my computer from malware?",
    "Explain how photosynthesis works.",
    "What are the benefits of regular sleep?",
]

THRESHOLDS = [
    0.80,
    0.85,
    0.90,
    0.95,
]

GEN_LENGTH = 64
BLOCK_LENGTH = 32
STEPS = 64

print("=" * 90)
print("SAFEDIFF ACCELERATION THRESHOLD SWEEP")
print("=" * 90)

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True
)

model = LLaDAModelLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(device).eval()

results = []

for threshold in THRESHOLDS:

    print("\n" + "-" * 80)
    print(f"THRESHOLD = {threshold}")
    print("-" * 80)

    total_time = 0.0
    total_nfe = 0

    for i, prompt in enumerate(PROMPTS):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output, nfe = generate_with_dual_cache(
                model,
                inputs.input_ids,
                steps=STEPS,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                threshold=threshold,
                            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_time += elapsed
        total_nfe += nfe

        decoded = tokenizer.batch_decode(
            output[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        print(
            f"{i+1:02d} | "
            f"time={elapsed:.3f}s | "
            f"nfe={nfe} | "
            f"{decoded[:100].replace(chr(10), ' ')}"
        )

    result = {
        "threshold": threshold,
        "total_time_sec": total_time,
        "average_time_sec": total_time / len(PROMPTS),
        "average_nfe": total_nfe / len(PROMPTS),
    }

    results.append(result)

    print(
        f"\nAVERAGE TIME: "
        f"{result['average_time_sec']:.4f}s"
    )

    print(
        f"AVERAGE NFE: "
        f"{result['average_nfe']:.2f}"
    )

with open(
    "results/acceleration_threshold_sweep.json",
    "w"
) as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 90)
print("SAVED:")
print("results/acceleration_threshold_sweep.json")
print("=" * 90)
