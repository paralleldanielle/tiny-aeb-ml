# Results

One row per checkpoint and split, from `scripts/eval.py` (greedy, at most 64 new tokens). Metric
definitions are in `docs/PLAN.md` under "Evaluation and logging". Rates are in %; TTC error is
the median absolute error in seconds on closing cases.

| Checkpoint | Split | Format | Decision acc. | Missed intervention | False alarm | TTC err (s) | TTC in tol. | Mean len (tok) | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
