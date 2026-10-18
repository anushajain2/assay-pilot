# Contributing

Thanks for considering it. AssayPilot is a small, deliberately narrow tool, and
the most useful contributions are usually the unglamorous ones: a failure case
it mishandles, an input format it should read, a sentence in a report that
overclaims.

## Getting set up

```bash
git clone https://github.com/assaypilot/assaypilot.git
cd assaypilot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[esm,app,dev]"
pytest
ruff check src tests && ruff format --check src tests
```

The test suite needs no model weights and no network: it runs on the
dependency-free physicochemical representation. If a change makes the suite
require a download, that is a design problem, not a test problem.

## The rules this codebase holds itself to

These are not style preferences. A change that breaks one of them will be asked
to change, however useful it is otherwise.

1. **Nothing is dropped or repaired silently.** Every row that does not reach
   the model gets an exclusion code, a reason, and its source row number.
   Mutation numbering is never adjusted to make an input fit.
2. **No leakage.** Identical sequences and their replicates stay in the same
   partition. Anything learnable — scaling, PCA, one-hot vocabulary,
   hyperparameters — is fitted inside training folds.
3. **A numerical lead is not a winner.** The selection rule prefers the simpler
   method unless the paired bootstrap interval excludes zero.
4. **Weak evidence is labelled.** If a result cannot carry a claim, the claim is
   not made — in the report, the exports, or the manifest.
5. **Observed, predicted, and selected are different categories.** They never
   merge into one column.
6. **A run can be reproduced from its manifest.** New configuration goes into
   `config`; runtime observations that do not affect results go into `runtime`,
   so the fingerprint stays stable.

[docs/claims.md](docs/claims.md) ties each of these to the code and the test
that enforces it. If you add a capability, add its claim — and its limits —
there.

## Making a change

- **Start with an issue** for anything beyond a typo, so the scope can be agreed
  before you write code.
- **Add the test that would have caught the bug**, named after the behaviour:
  `test_diversity_policy_reports_shortfall_without_relaxing`, not `test_select_2`.
- **Write for the next reader.** Comments explain *why*, especially where the
  honest choice is not the obvious one. Module docstrings say what the module
  is for and what it refuses to do.
- **Keep the public surface small.** The CLI, the Streamlit interface, and the
  benchmark harness all call `pipeline.plan_batch`; there should stay exactly
  one implementation of the decision.
- **User-facing errors are written for a researcher**, not for a stack trace
  reader: say which input was wrong and what would make it right.

## Especially welcome

- **Failure cases.** A real input that AssayPilot mishandles, misreports, or
  rejects for the wrong reason. A CSV that reproduces it is a gift.
- **Public-dataset replays.** `assaypilot benchmark` exists so the decision
  process can be measured. Results — including ones where nothing beat the
  baseline — are valuable. See [docs/benchmarks.md](docs/benchmarks.md).
- **Overclaiming.** Any sentence in a report, docstring, or README that asserts
  more than the evidence supports.
- **Scope items from the roadmap**: indels, multi-objective selection,
  calibrated intervals with measured coverage, additional representations.

## Things that need discussion first

- **Adding a method to the comparison suite.** The suite is small on purpose;
  searching across many models until one wins inflates apparent performance on
  small assay datasets. A new method needs a reason it answers a question the
  existing three cannot.
- **Uncertainty-based selection.** Version one leaves it out deliberately.
  Reintroducing it needs measured calibration for the relevant setting, not an
  argument that it ought to work.
- **New required dependencies.** `torch` and `streamlit` are optional extras
  for good reasons.

## Pull requests

Keep them focused, explain the behaviour change in the description, and say
which test demonstrates it. CI runs the suite on Linux and macOS across
supported Python versions, plus `ruff`. Green CI and a clear description are
usually enough.

By contributing you agree that your contribution is licensed under Apache-2.0,
the same as the project.
