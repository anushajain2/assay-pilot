# AssayPilot documentation

Start with the [project README](../README.md) for what AssayPilot is and how to
run it. These pages go deeper.

| Page | Read it when you want to know |
| --- | --- |
| [claims.md](claims.md) | Exactly what the software asserts, what it refuses to assert, and which code and test enforce each line. Read this before quoting any AssayPilot output in a paper or a meeting. |
| [methodology.md](methodology.md) | Why the splits, metrics, bootstrap, and selection rule are what they are — the reasoning behind each decision in the chain. |
| [data-formats.md](data-formats.md) | Every input column and every output file, field by field, including the run manifest. |
| [representations.md](representations.md) | One-hot, ESM-2, and the dependency-free physicochemical features; the embedding cache; how to plug in your own representation. |
| [benchmarks.md](benchmarks.md) | How to evaluate the whole decision process on public datasets, and the reporting rules that keep the result honest. |
| [testing.md](testing.md) | What each test module protects against, and how to add a test. |

## A quick map of the code

| Question | File |
| --- | --- |
| What is a campaign? | [`campaign.py`](../src/assaypilot/campaign.py) |
| How are inputs validated? | [`data/loaders.py`](../src/assaypilot/data/loaders.py), [`data/validation.py`](../src/assaypilot/data/validation.py) |
| How is the evaluation protocol chosen? | [`evaluation/splits.py`](../src/assaypilot/evaluation/splits.py) |
| How is a method chosen? | [`evaluation/compare.py`](../src/assaypilot/evaluation/compare.py) |
| How is a batch selected? | [`selection/filters.py`](../src/assaypilot/selection/filters.py), [`selection/policies.py`](../src/assaypilot/selection/policies.py) |
| What ties it together? | [`pipeline.py`](../src/assaypilot/pipeline.py) |
| What is recorded? | [`reporting/manifest.py`](../src/assaypilot/reporting/manifest.py) |

## Conventions in this codebase

- **Nothing is dropped or repaired silently.** Every exclusion carries a code,
  a reason, and a source row.
- **Predictions are desirability.** Higher is always better; the pipeline flips
  the sign once for a minimisation objective, so ranking code never has to
  think about direction.
- **A method describes its own limits.** `MethodInfo.caveats` holds the
  sentences that must appear beside its numbers.
- **The one implementation of the decision is `pipeline.plan_batch`.** The CLI,
  the interface, and the benchmark harness all call it.
