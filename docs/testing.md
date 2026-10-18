# What the tests protect

```bash
pytest                 # whole suite, no model weights required
pytest tests/test_leakage.py -v
ruff check src tests && ruff format --check src tests
```

The suite runs on the dependency-free physicochemical representation, so it
needs no torch and no network. ESM-2 paths are exercised through availability
reporting rather than by downloading a checkpoint.

| Module | What breaks if it fails |
| --- | --- |
| `test_sequences.py` | Mutation notation is misread, or a mismatched wild-type residue is silently renumbered — which would reinterpret an entire dataset against the wrong positions. |
| `test_loaders.py` | Rows are dropped without a reason, foreign-condition measurements are pooled, censored values are treated as exact, replicate disagreement is hidden, or a log transform silently offsets non-positive values. |
| `test_splits.py` | The wrong evaluation protocol is chosen, a random split is presented as extrapolation evidence, or a variant remeasured in a later round sits on both sides of a split. |
| `test_leakage.py` | A method sees held-out rows while fitting; the scaler or PCA is fitted over the whole dataset; the one-hot vocabulary learns substitutions only present in held-out data. These are the failures that inflate every number downstream. |
| `test_compare.py` | A small numerical lead becomes a recommendation; weak evidence is not labelled; the comparison is not reproducible. |
| `test_selection.py` | A locked position or substitution limit is violated; an already-measured sequence is re-ordered by accident; the diversity constraint is quietly relaxed to fill a batch; coverage warnings are not raised; rank variability is described as a probability. |
| `test_pipeline.py` | The batch, the manifest, or the exports lose provenance; a rerun is not deterministic; `design_id` drifts; observed and predicted values get mixed. |
| `test_rounds.py` | Results cannot be matched back to predictions; adding a round edits an old plan; the store loses history. |
| `test_cli.py` | Broken input is accepted without `--skip-invalid`; exit codes are wrong; the demo stops working. |
| `test_replay.py` | The benchmark gives strategies different starting points, exceeds the budget, re-measures a variant, or drops the caveats that bound what a replay can show. |

## Conventions

- **Deterministic.** Every fixture has a fixed seed; no test depends on wall
  time, network, or an installed checkpoint.
- **Small.** The shared simulated campaign is 90 measurements and 120
  candidates, which is enough to exercise rounds, replicates, and coverage.
- **Behavioural, not structural.** Tests assert on what a researcher would see
  — exclusion codes, recommendation text, exported columns — rather than on
  internal call shapes, so refactors stay cheap.
- **Test doubles for the decision rule.** `ConstantMethod` and
  `RecordingMethod` let the comparison logic be tested against known-good and
  known-useless predictors, which real models cannot guarantee.

## Adding a test

If you fix a bug, add the test that would have caught it, and name it after the
behaviour rather than the function:
`test_diversity_policy_reports_shortfall_without_relaxing`, not
`test_select_2`.

If you add a claim to the software, add it to [claims.md](claims.md) with the
test that enforces it.
