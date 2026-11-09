# Python style

Read before writing or reviewing code. Adapted from NVIDIA's open-source Python guidelines
(TensorRT-LLM, Megatron-Bridge, NeMo, BioNeMo), which build on PEP 8 and the Google Python Style
Guide.

**Don't fight the tooling.** `ruff format` (line length 100), `ruff check` (lint, import order,
Google docstrings) and `mypy` on `src/` are configured in `pyproject.toml`. If they accept it,
that is the style. The rules below are the ones the tools can't check.

No per-file license headers: the root `LICENSE` (Apache-2.0) covers the repo.

## Naming

| Kind | Convention | Example |
| --- | --- | --- |
| Files, modules | snake_case | `make_data.py` |
| Classes | PascalCase | `ScenarioSample` |
| Functions, methods, locals | snake_case | `compute_ttc` |
| Constants (any scope) | UPPER_SNAKE_CASE | `WARN_TTC_S = 5.0` |
| Name would start with a digit | prefix `k` | `k_99th_percentile` |
| Not public | leading underscore | `_parse_line` |

- Private (`_`) functions are not called outside their own file.
- Single uppercase letters are fine for math (`B`, `T`, `N`, `V`).
- Don't shadow builtins or names from an outer scope. Initialize every public attribute in
  `__init__`. No mutable global state.
- No ad-hoc abbreviations (`bn` → `batch_norm`). Established ones are fine: `ttc`, `aeb`, `lr`,
  `kl`, `logits`, `logprobs`. Use the names the literature and TRL use (`num_generations`,
  `hidden_size`).
- Physical quantities carry their SI unit in the name: `distance_m`, `speed_mps`, `ttc_s`.
  Convert to SI at the boundary; `_kph` names appear only in `templates.py`.

## Imports

- No wildcard imports. Every module in `src/` defines `__all__`.
- Import the module when its name is unique, then use it qualified:
  `from tiny_aeb_ml import physics` → `physics.compute_ttc(...)`.

## Type hints

- Annotate every function, including `-> None`, and every dataclass field.
- Built-in generics and `|`: `list[int]`, `float | None`. Not `List`, `Optional`, `Union`.
- `Literal` for closed sets: `Decision = Literal["NONE", "WARN", "BRAKE", "EMERGENCY"]`.
- Typed callables: `Callable[[str], float]`, not `Callable[..., float]`.
- Avoid `Any` and `# type: ignore`. Untyped third-party libraries are handled in the mypy config,
  not inline.

## Docstrings and comments

- Google-style docstrings on every public module, class and function: what it does and why you'd
  use it. Document constructor arguments.
- Tensor arguments state shape and dtype:

```python
def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum the log-probabilities of the target tokens for each sequence.

    Args:
        logits: Model output, shape [batch, seq_len, vocab], float32.
        labels: Target token ids, shape [batch, seq_len], int64; -100 marks ignored positions.

    Returns:
        Per-sequence log-probability, shape [batch], float32.
    """
```

- Comments explain why, in complete sentences. A long function divided by comments wants helper
  functions instead.
- Commented-out code either carries a comment saying why it is disabled, or gets deleted.
- Learning-project rule: where a line does something non-obvious about LLM training (loss
  masking, EOS handling, advantage normalization, the KL term), add a short "why" comment even if
  an expert wouldn't need it.

## Error handling

- Raise built-in exceptions (`ValueError`, `TypeError`, `FileNotFoundError`) instead of returning
  error values. No `assert` for runtime checks outside tests.
- Catch the narrowest type. Keep `try` bodies minimal and put the follow-up logic in `else`.
- Prefer `isinstance()` to duck typing.
- Exception: `parse_completion()` returns `None` for unparseable model output. That is an
  expected outcome the reward scores, not an error.

## Configs

- Config objects are `@dataclass(frozen=True)` loaded from `configs/*.yaml`, named with a
  `Config` suffix (`DataConfig`, `RunConfig`). Don't reuse TRL's names (`SFTConfig`,
  `GRPOConfig`).
- No arbitrary defaults: anything that changes results (learning rate, seed, steps, group size)
  is required in the YAML.
- TTC thresholds are constants in `physics.py`, not config.

## Logging

- `logging`, not `print`, inside `src/`. Scripts configure logging once (INFO) in `main()`;
  `print` is fine for the final summary a script shows.

## Testing

- Test files mirror modules: `tests/test_physics.py`, `tests/test_format.py`. Use fixtures for
  shared setup.
- Unit tests are fast and need no model download. Anything that loads a model or trains is marked
  `@pytest.mark.slow`.
- A bug fix comes with a test that fails without the fix.
