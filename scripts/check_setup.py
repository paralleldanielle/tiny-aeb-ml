"""M0 setup check: load SmolLM2-135M on CPU, generate a few tokens and report the cost.

Run with ``uv run python scripts/check_setup.py``. The first run downloads the model (about
270 MB) into the Hugging Face cache.
"""

import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_aeb_ml import runtime

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "Ego speed 72 km/h. Car ahead at 38 m, driving 40 km/h. Decision?"
MAX_NEW_TOKENS = 30

logger = logging.getLogger(__name__)


def main() -> None:
    """Load the base model, greedily continue one prompt and print timings and peak RAM."""
    runtime.configure_logging()
    num_threads = runtime.configure_threads()
    for package, version in runtime.package_versions().items():
        logger.info("%s %s", package, version)
    device = torch.device("cpu")

    start_s = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    model.eval()
    load_s = time.perf_counter() - start_s
    num_params = sum(param.numel() for param in model.parameters())

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    start_s = time.perf_counter()
    with torch.inference_mode():
        # Greedy decoding (no sampling) makes the output reproducible. The base model has no
        # pad token; reusing EOS as pad silences a warning and changes nothing for one prompt.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generate_s = time.perf_counter() - start_s

    # generate() returns prompt + continuation; slice off the prompt to show only new tokens.
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0, prompt_len:]
    continuation = tokenizer.decode(new_ids, skip_special_tokens=False)

    print(f"model:        {MODEL_ID} ({num_params / 1e6:.1f}M params, fp32)")
    print(f"threads:      {num_threads}")
    print(f"prompt:       {PROMPT!r} ({prompt_len} tokens)")
    print(f"continuation: {continuation!r}")
    print(f"load time:    {load_s:.1f} s")
    print(
        f"generate:     {len(new_ids)} tokens in {generate_s:.2f} s "
        f"({len(new_ids) / generate_s:.1f} tokens/s)"
    )
    print(f"peak RAM:     {runtime.peak_rss_mb():.0f} MiB")


if __name__ == "__main__":
    main()
