# AGENTS.md — tiny-aeb-ml

A learning project: post-train SmolLM2-135M on a CPU-only machine to make toy AEB decisions
(`NONE` / `WARN` / `BRAKE` / `EMERGENCY`) from a short text scene. The goal is understanding, not a
good model.

The owner is an experienced systems engineer (Active Safety, AD integration) who is new to LLM
post-training. Explain LLM-specific concepts briefly the first time they come up; don't explain
Python, git or basic ML. Prefer readable, flat scripts (one per stage, top to bottom) over
abstractions.

## Where things are

- `docs/PLAN.md`: what to build (task spec, data, milestones, experiments, eval). Its
  **Status** line says which milestone is next.
- `docs/STYLE.md`: how to write Python. Read it before writing or reviewing code.
- `docs/notes/LEARNING_LOG.md`, `docs/notes/RESULTS.md`: what has run and what it measured.

On what to build, `PLAN.md` wins; on how to work, this file wins.

## Commands

```bash
uv sync                                     # install pinned deps into .venv
uv run pytest                               # fast tests (slow ones excluded)
uv run pytest -m slow                       # tests that load a model or train
uv run ruff format . && uv run ruff check --fix .
uv run mypy src
uv run python scripts/check_setup.py        # load the model, generate, report cost
```

Always go through `uv run`. Format, lint, mypy and fast tests pass before every commit.

## Hard rules

- **CPU only.** No `.cuda()`, `device_map="auto"`, vLLM, bitsandbytes, flash-attention,
  DeepSpeed or Unsloth. Never hard-code a device in `src/`; scripts pass `torch.device("cpu")`
  down.
- fp32. bf16 only after a measured test on this machine shows it is faster and correct.
- RAM is 6.7 GiB total. Check peak RAM against that before scaling batch, sequence length or
  model size.
- Script entry points call `runtime.configure_threads()` (physical cores, logged) and seed
  `random`, `numpy`, `torch` and `transformers.set_seed` from the config.
- Every training script has `--smoke` (about 10 steps on tiny data, under 2 minutes). Run it
  before any long run.
- Every real run writes its resolved config, package versions, git commit, wall-clock time and
  peak RAM into its run directory.
- Ask before adding a dependency that is not in `pyproject.toml`.
- TRL and transformers argument names and defaults drift between releases. Check the installed
  version (`dataclasses.fields(GRPOConfig)`, `inspect.signature`) instead of memory or tutorials.
- Don't silently "fix" a bad experimental result. Explain it in the learning log; bad results are
  often the lesson.
- `data/` and `runs/` are gitignored. Never commit checkpoints, datasets or large logs.
- The repo is public: no personal names or employer details in any file.

## After each milestone

Append an entry of at most 15 lines to `docs/notes/LEARNING_LOG.md` (what ran, key numbers, what
was surprising, where to look in the code), add eval rows to `docs/notes/RESULTS.md`, update the
Status line in `docs/PLAN.md`, and commit it all with the code. One commit per milestone or
logical change; imperative subject under 72 characters.
