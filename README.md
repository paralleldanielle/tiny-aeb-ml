# tiny-aeb-ml

A learning project: post-train [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
on a CPU-only machine to make toy AEB (automatic emergency braking) decisions from a short text
scene, using SFT and then GRPO.

```text
Ego speed 72 km/h. Car ahead at 38 m, driving 40 km/h. Decision?
```

```text
v_ego=20.0 m/s
v_lead=11.1 m/s
v_rel=8.9 m/s
TTC=38.0/8.9=4.3 s
DECISION: WARN
```

The thresholds are toy values, not a real AEB calibration.

## Quickstart

```bash
uv sync
uv run pytest
uv run python scripts/check_setup.py
```

## Docs

- [`docs/PLAN.md`](docs/PLAN.md): task spec, milestones, experiments
- [`docs/MODEL.md`](docs/MODEL.md): SmolLM2-135M architecture, diagram and walkthrough
- [`docs/notes/LEARNING_LOG.md`](docs/notes/LEARNING_LOG.md): what each milestone taught
- [`docs/notes/RESULTS.md`](docs/notes/RESULTS.md): eval results per checkpoint and split
- [`AGENTS.md`](AGENTS.md) and [`docs/STYLE.md`](docs/STYLE.md): rules for coding agents
