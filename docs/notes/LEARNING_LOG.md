# Learning log

One entry per milestone, at most 15 lines: what ran, key numbers, what was surprising, where to
look in the code. Experiments get 3–5 lines each.

## M0 — Setup (2026-11-09)

- Ran `scripts/check_setup.py`: load SmolLM2-135M in fp32, greedily generate 30 tokens.
- Stack: Python 3.12.16, torch 2.15.0+cpu, transformers 5.22.2, TRL 1.17.1, datasets 5.1.0.
- 134.5M parameters. The 28-token prompt generates at ~22 tokens/s on 6 threads. Load takes
  1.7 s from cache (50 s on the first run, which includes the ~270 MB download). Peak RAM is
  1.2 GiB for inference alone.
- Output: `"\n\nThe car is going 40 km/h.\n\nThe car is going 40 km/h.\n\nThe car"`. A base
  model only continues text the way web text tends to continue; it has no idea there is a task.
  Greedy decoding always takes the most likely next token, and once a phrase has appeared,
  repeating it becomes the most likely continuation, hence the loop. SFT teaches the format;
  ending each target with EOS teaches it to stop.
- Cost intuition for M4: one GRPO prompt with 8 completions × 64 tokens is ~512 generated
  tokens, about 23 s unbatched at this speed. Batched generation will matter.
- Gotchas: transformers 5 renamed `torch_dtype` to `dtype`; torch already defaults to 6 threads
  here; HF Hub logs every HTTP request on the `httpx2` logger (silenced in
  `runtime.configure_logging`).
