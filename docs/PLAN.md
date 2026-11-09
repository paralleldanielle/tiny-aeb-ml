# Tiny AEB ML — Plan

What to build: task spec, data, milestones, experiments, evaluation. How to work (rules, commands)
is in `AGENTS.md`; how to write Python is in `docs/STYLE.md`.

## Goal

Teach a 135M-parameter open language model a toy AEB (automatic emergency braking) decision task,
on a CPU-only machine, in about two afternoons. The point is learning, not a good model: hands-on
understanding of small-LM architecture, training data, SFT and RL post-training (GRPO, optionally
DPO).

- **The task:** a short text scene goes in (ego speed, distance and speed of the object ahead). A
  short worked calculation and a decision come out: `NONE`, `WARN`, `BRAKE` or `EMERGENCY`. Python
  computes the ground truth, so data is free and rewards are exact.
- **Why this task:** it touches every stage in a domain the owner knows: number tokenization,
  synthetic data and distribution shift, SFT, verifiable-reward RL, reward hacking, and asymmetric
  safety costs.
- **Non-goals:** a realistic AEB function, benchmark scores, GPU support, serving, quantization.

## Decisions

Settled at M0 (2026-11-09):

- **Machine:** Intel i5-12400, 6 physical cores / 12 threads, AVX2 and AVX-VNNI but no AVX-512 or
  AMX, Ubuntu 26.04. Use 6 threads and fp32.
- **RAM is the binding constraint:** 6.7 GiB total plus 4 GiB swap. A full fine-tune of 135M
  (about 2–3 GB) fits with other apps closed; 360M (6–8 GB) does not.
- **Repo:** `tiny-aeb-ml`, public on GitHub; Python package `tiny_aeb_ml`.
- **German test split:** skipped in v1. It can be added later as an extra eval split without
  touching training.
- **TTC thresholds:** the toy values below.

## Stack

| Item | Choice |
| --- | --- |
| Python | 3.12 (3.11+), managed with `uv` |
| Core stack | `torch` (CPU wheel), `transformers`, `datasets`, `trl`, `accelerate`, `pyyaml` |
| Add when needed | `matplotlib` (M2 plots), `pandas`, `peft` (M6 LoRA only) |
| Dev | `ruff`, `mypy`, `pytest` |
| Base model | `HuggingFaceTB/SmolLM2-135M` (base, not Instruct) |
| Precision | fp32 |
| Device | `use_cpu=True` in the training args |
| Threads | `torch.set_num_threads(6)`, logged at startup |

Memory for a full fine-tune with AdamW is about 16 bytes per parameter plus activations.

Settings that keep CPU runs short:

- Max sequence length about 192 tokens; prompts and answers here are short.
- Small per-device batch plus gradient accumulation.
- GRPO: `max_completion_length` at most 64, `num_generations` 4–8, `use_vllm=False`. Generation is
  the bottleneck, not the backward pass.
- Rough expectations: SFT on 5k short examples on the order of an hour, a few hundred GRPO steps
  in 1–2 hours. Measure on the first smoke run and scale data or steps to fit an afternoon.

## Task spec

All physics and thresholds live in one module, `src/tiny_aeb_ml/physics.py`; data generation,
rewards and eval import from it. The thresholds are toy values, not a real AEB calibration.

Closing speed is v\_rel = v\_ego − v\_lead in m/s (stationary object: v\_lead = 0). If v\_rel ≤ 0
there is no collision course: TTC = inf and the decision is `NONE`. Otherwise:

```latex
\mathrm{TTC} = \frac{d}{v_{\mathrm{ego}} - v_{\mathrm{lead}}}
```

| Decision | TTC (s) |
| --- | --- |
| `NONE` | ≥ 5.0, or not closing |
| `WARN` | 2.5 to < 5.0 |
| `BRAKE` | 1.0 to < 2.5 |
| `EMERGENCY` | < 1.0 |

The ground-truth decision uses the unrounded TTC. Displayed numbers are rounded to 1 decimal; the
TTC tolerance in the reward absorbs that. Near a threshold the displayed TTC can land on the other
side of it from the decision (`TTC=…=2.5 s`, then `DECISION: BRAKE`). That is expected, not a bug.

**Prompt** (one of many templates, see Data):

```text
Ego speed 72 km/h. Car ahead at 38 m, driving 40 km/h. Decision?
```

**Completion** (fixed five-line format, ends with EOS):

```text
v_ego=20.0 m/s
v_lead=11.1 m/s
v_rel=8.9 m/s
TTC=38.0/8.9=4.3 s
DECISION: WARN
```

Not closing (ego 50 km/h, lead 70 km/h): `v_rel=-5.6 m/s`, then `TTC=inf`, then
`DECISION: NONE`.

One parser, `parse_completion(text) -> ParsedCompletion | None` in `format.py`, is used by rewards
and eval and is unit-tested.

**Rewards:** separate TRL reward functions with signature `(completions, **kwargs) -> list[float]`.
Dataset columns arrive as kwargs, so ground truth never has to be re-parsed from the prompt.
Maximum total is 1.0.

| Function | Value |
| --- | --- |
| `format_reward` | 0.2 if all five lines parse in order, else 0 |
| `ttc_reward` | 0.3 if TTC is within 10 % or 0.2 s of truth, whichever is larger (inf matches inf), else 0 |
| `decision_reward` | 0.5 if the decision is correct, else 0 |
| `safety_reward` (experiment E5 only) | −1.0 if the predicted decision is less severe than truth, −0.2 if more severe, else 0 |

## Data generation and splits

`scripts/01_make_data.py` writes seeded JSONL files with `prompt`, `completion` and the
ground-truth columns `distance_m`, `ego_speed_mps`, `lead_speed_mps`, `ttc_s`, `decision`,
`template_id`, `speed_unit`, `object_type`.

Sampling:

- Ego speed 10–130 km/h in train (130–160 is reserved for the OOD split). Distance 2–150 m.
- Object: car, truck, cyclist or pedestrian, about 15 % stationary. Lead speed is plausible per
  object: pedestrian 0–7 km/h, cyclist 0–30 km/h, car or truck 0 to ego + 30 km/h.
- Speeds given in km/h or m/s at random; distance always in m.
- Rebalance by rejection sampling so each decision class is 20–30 % of train. Uniform sampling
  would make `NONE` dominate.
- About 20 English prompt templates; 5 of them are held out entirely.

| Split | Examples | What it tests |
| --- | --- | --- |
| `train` | 5,000 | Training data |
| `val` | 300 | Same distribution, used during training |
| `test_iid` | 500 | Same distribution as train |
| `test_ood_speed` | 500 | Ego speed 130–160 km/h, never seen in training |
| `test_ood_template` | 500 | The 5 held-out phrasings |

The script prints, and the owner reviews before training: class balance per split, histograms of
TTC and speeds (PNG), 5 random examples per split, token-length stats with the SmolLM2 tokenizer,
and an explicit check that no OOD condition leaks into train.

GRPO uses train prompts only (its completion column is ignored); a fresh seeded sample of about
2,000 prompts is fine.

## Repo layout

```text
tiny-aeb-ml/
├── AGENTS.md                   # agent rules, always loaded
├── README.md
├── pyproject.toml              # uv; exact pins, tool config
├── configs/
│   ├── sft.yaml
│   └── grpo.yaml               # experiments = small overrides of this
├── docs/
│   ├── PLAN.md                 # this file
│   ├── STYLE.md                # Python style
│   └── notes/
│       ├── LEARNING_LOG.md     # one entry per milestone
│       └── RESULTS.md          # one row per checkpoint x split
├── src/tiny_aeb_ml/
│   ├── runtime.py              # threads, versions, peak RAM for script entry points
│   ├── physics.py              # TTC, thresholds, decision: single source of truth
│   ├── templates.py            # prompt phrasings, train vs held-out
│   ├── format.py               # build target completion + parse_completion()
│   ├── rewards.py              # TRL reward functions
│   └── evaluate.py             # metrics, confusion matrix
├── scripts/
│   ├── check_setup.py          # M0
│   ├── 00_inspect_model.py
│   ├── 01_make_data.py
│   ├── 02_sft.py
│   ├── 03_grpo.py
│   ├── 04_dpo.py               # stretch
│   ├── 05_grpo_from_scratch.py # stretch
│   ├── eval.py                 # any checkpoint x any split
│   └── plot_run.py
├── tests/                      # mirrors src/: physics, format, rewards, ...
├── data/                       # gitignored
└── runs/                       # gitignored: checkpoints, logs, plots
```

## Milestones

Afternoon 1 covers M0–M3, afternoon 2 covers M4–M5, and M6 is an optional third session. Times
are rough targets, not limits.

**Status:** M0 done. Next: M1.

### M0 — Setup (about 20 min)

- `uv` project, pinned deps, CPU torch. `scripts/check_setup.py` loads SmolLM2-135M, generates 30
  tokens from a test prompt, and prints versions, thread count, wall time and peak RAM.
- Done when: `pytest` runs and the model generates text.

### M1 — Inspect the architecture (about 30 min) — `00_inspect_model.py`

- Print the config. Break parameters down into embedding, attention, MLP and norms; confirm the
  input and output embeddings are tied (about 28M of the 135M are the embedding table).
- Point to where GQA, RoPE, RMSNorm and SwiGLU live in the installed transformers modeling code
  (file and function names). Print the attention-head and KV-head counts.
- Tokenize sample prompts and completions; show how `72`, `38.0`, `4.3`, `km/h` and `m/s` split
  into tokens.
- Done when: the breakdown sums to the total parameter count and the log entry explains the
  embedding share.

### M2 — Data (about 45 min) — `physics.py`, `templates.py`, `format.py`, `01_make_data.py`

- Done when: unit tests pass (not closing, stationary object, TTC exactly on each threshold,
  km/h vs m/s, parser round-trip), all splits are written, and the owner has reviewed the
  sanity-check output.

### M3 — Baseline and SFT (about 1.5 h including training) — `eval.py`, `02_sft.py`

- Baseline: evaluate the base model with a 3-shot prompt on `test_iid`. Expect close to zero.
- Full fine-tune in prompt–completion format with loss on the completion only. Print one
  tokenized example showing which tokens are masked, and confirm the completion ends with EOS
  (otherwise GRPO completions won't stop).
- Starting point: learning rate 5e-5 with cosine decay, 1–2 epochs, effective batch 32.
- Done when: eval on all splits is in `docs/notes/RESULTS.md` and the checkpoint is in
  `runs/sft/`.

### M4 — GRPO (about 1.5 h) — `03_grpo.py`

- Start from `runs/sft/`. Rewards: format + TTC + decision. `num_generations` 8,
  `max_completion_length` 64, temperature 0.8–1.0, learning rate 1e-6 to 5e-6, a few hundred
  steps.
- Set `beta` (KL weight) explicitly and log what the installed TRL version defaults to. A
  non-zero `beta` loads a frozen reference model, which costs another ~0.5 GB of RAM.
- Done when: reward curves are plotted, all splits are evaluated against the SFT checkpoint, and
  the log entry explains what changed, especially on the OOD splits.

### M5 — Experiments (rest of afternoon 2)

- E1–E5 from the next section. Each is a config override, not a new script.

### M6 — Stretch (optional third session)

- `04_dpo.py`: sample 4 completions per prompt from the SFT model, pair a correct one (chosen)
  with an incorrect one (rejected), train DPO, compare with GRPO at similar compute.
- `05_grpo_from_scratch.py`: about 150 lines of plain PyTorch. Sample a group, score it, compute
  advantages as (r − mean) / std, apply the clipped token-level policy loss, optional KL to a
  frozen reference. Check the reward curve roughly tracks TRL's on the same config.
- A bigger model: SmolLM2-360M with LoRA (a full fine-tune doesn't fit in RAM here). Compare
  accuracy and wall-clock time with 135M.

## Experiments

Each experiment gets its own run directory, the same eval as M4, and a 3–5 line learning-log
entry with its plots. Expected outcomes are hypotheses to check, not results to force.

| ID | Change | Likely outcome and lesson |
| --- | --- | --- |
| E1 | GRPO from the SFT checkpoint (the M4 run) | Reference run. Compare gains on IID vs. OOD splits |
| E2 | GRPO from the base model, no SFT | Likely flat: when every completion in a group gets the same reward, the advantage is zero and nothing is learned. Log the share of zero-variance groups |
| E3 | Format reward only | Format reaches about 100 % while TTC and decision stay flat or degrade: reward hacking in miniature |
| E4 | Learning rate x10 | Outputs collapse toward one answer: entropy drops, completion length shifts, reward stalls |
| E5 | Add `safety_reward` | Model shifts toward over-warning. Compare missed-intervention rate and false-alarm rate with E1 |

## Evaluation and logging

`scripts/eval.py --ckpt <path> --split <name>` decodes greedily (temperature 0, at most 64 new
tokens) and writes `runs/<run>/eval_<split>.json`. Every checkpoint is evaluated on every split,
and one row per checkpoint and split goes into `docs/notes/RESULTS.md`.

Metrics per split:

- Format validity: % of completions that parse.
- Decision accuracy and the 4x4 confusion matrix.
- Missed-intervention rate: truth is `BRAKE` or `EMERGENCY`, prediction is `NONE` or `WARN`. This
  is the safety-relevant number.
- False-alarm rate: truth is `NONE`, prediction is `BRAKE` or `EMERGENCY`.
- TTC error on closing cases: median absolute error and % within tolerance.
- Mean completion length in tokens.

Training logs: save `trainer.state.log_history` to JSON after every run; no wandb. `plot_run.py`
turns it into PNGs in the run directory. For GRPO, plot at least total reward, each reward
function separately, reward std, KL, completion length, and entropy if the installed version logs
it.

## References

- [TRL GRPO Trainer docs](https://huggingface.co/docs/trl/en/grpo_trainer) and
  [TRL quickstart](https://huggingface.co/docs/trl/quickstart) (SFT, DPO, GRPO)
- [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [SmolLM2 paper](https://arxiv.org/abs/2502.02737) (data and training recipe)
- [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) (introduces GRPO)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [TRL + GRPO end-to-end on CPU with GPT-2](https://varun1724.github.io/rlwatch/tutorials/trl-grpo-end-to-end/)
  (proof that GRPO runs on CPU)
- [Hugging Face cookbook: GRPO with TRL](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl)
