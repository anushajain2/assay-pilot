# AssayPilot

[![CI](https://github.com/anushajain2/assay-pilot/actions/workflows/ci.yml/badge.svg)](https://github.com/anushajain2/assay-pilot/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Plan the next round of protein variant experiments from the measurements you already have.**

You measured 150 enzyme variants. You have a pool of 1,000 untested ones and room for 24 wells
next week. Which 24, and what evidence supports that choice?

AssayPilot answers that question and shows its work. It checks your data, compares a small fixed
set of prediction methods on *your* assay using held-out measurements, picks the method the
evidence actually supports, selects a batch under your constraints, and exports a plan you can
hand to a colleague — a CSV, a FASTA, a readable HTML report, and a machine-readable manifest
that records every input checksum, seed, and hyperparameter behind the decision.

It runs locally. Your measurements are never sent anywhere.

```bash
pip install -e ".[esm]"              # from a checkout; see Install
assaypilot demo -o demo_output       # whole workflow on simulated data, no downloads
```

---

## Contents

- [Why this exists](#why-this-exists)
- [What it does, concretely](#what-it-does-concretely)
- [Install](#install)
- [Quickstart](#quickstart)
- [What a run actually says](#what-a-run-actually-says)
- [Input formats](#input-formats)
- [How the decision is made](#how-the-decision-is-made)
- [What comes out](#what-comes-out)
- [The round loop](#the-round-loop)
- [Command reference](#command-reference)
- [Python API](#python-api)
- [Scope of version one](#scope-of-version-one)
- [What AssayPilot claims, and what it does not](#what-assaypilot-claims-and-what-it-does-not)
- [How it relates to existing tools](#how-it-relates-to-existing-tools)
- [Documentation](#documentation)
- [Project layout](#project-layout)
- [Development](#development)
- [Status](#status)
- [References](#references)
- [License and citation](#license-and-citation)

---

## Why this exists

Three things make the "which variants next?" decision harder than it looks.

**1. The decision happens at the assay level.** A protein can be evaluated for expression,
catalytic activity, or activity retained after heating. These are different outcomes, and
improving one does not guarantee improving another. A protein language model's representation
may be useful across tasks, but the relationship between that representation and *your* measured
outcome has to be established for *your* task. AssayPilot ties every trained predictor to a named
measurement under a named condition. It never reinterprets a cellular readout as an intrinsic
catalytic constant, and never treats sequence plausibility as a measurement of function.

**2. Public benchmarks cannot answer a local question.** ProteinGym and FLIP are essential for
choosing credible starting points, and this project uses them. But no benchmark can establish
performance on a new lab's measurements or on its particular next set of candidates. Those
questions need held-out data from your own campaign.

| Question | Evidence that answers it |
| --- | --- |
| Which methods are credible starting points? | Public benchmarks and published methods |
| Does a method predict *this* assay's outcome? | Held-out measurements from this campaign |
| Can it extrapolate to *these* candidates? | Evaluation matching their mutation distance or round, plus coverage checks |
| Did the chosen batch improve the experiment? | Measured results from the next round |

**3. A small local test is not automatically better evidence than a large benchmark.** If the
local estimate is unstable, the honest answer is "there is not enough evidence to distinguish
these methods" — and AssayPilot gives that answer instead of promoting a 0.02 lead in rank
correlation into a winner.

---

## What it does, concretely

- **Catches input problems before they reach a batch**: sequence/reference mismatches, mutation
  notation that contradicts the reference, duplicate candidates, measurements from a different
  condition, censored values, replicate disagreement, and train/test leakage through replicates.
- **Compares three methods on the same data and the same budget**: a simple baseline, a
  label-free pretrained score, and an assay-trained model on frozen embeddings.
- **Evaluates the way you will actually use the model**: earlier rounds predicting a later round
  when round labels exist; otherwise holdouts built to resemble the candidate pool. A random
  split is reported as an interpolation diagnostic, never as evidence about distant candidates.
- **Selects a batch deterministically** under locked positions, substitution limits, and an
  optional diversity constraint, recording every exclusion and every reason.
- **Preserves the evidence across rounds**: when the next round's results arrive, it scores the
  previous predictions *before* retraining, then versions the campaign. Old plans never change.

---

## Install

Python 3.10 or newer. AssayPilot is not on PyPI yet, so install from a
checkout:

```bash
git clone https://github.com/anushajain2/assay-pilot.git
cd assay-pilot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[esm,app,dev]"
pytest
```

The extras are independent, so install only what you need:

| Extra | Adds |
| --- | --- |
| *(none)* | Validation, the simple baseline, selection, exports, CLI |
| `esm` | ESM-2 embeddings and masked-marginal scores (torch + transformers) |
| `app` | The Streamlit interface |
| `dev` | Test suite and linter |

**Without the `esm` extra**, AssayPilot still runs end to end: the pretrained ESM-2 score is
reported as unavailable (with the reason), and the assay-trained model uses a dependency-free
physicochemical representation that is labelled, everywhere it appears, as *not* a protein
language model. This keeps tests, CI, and a first look on a laptop free of model downloads.

---

## Quickstart

### 1. See it work on simulated data

```bash
assaypilot demo -o demo_output
open demo_output/plan/report.html
```

This generates a synthetic campaign (150 measured variants across 3 rounds, 1,000 candidates),
runs the full workflow, and writes the exports. The data is simulated by a documented generative
model and says nothing about any real protein — every artifact is labelled accordingly.

A checked-in copy of the same kind of campaign lives in
[`examples/simulated-enzyme/`](examples/simulated-enzyme/), so you can look at the input files
before preparing your own:

```bash
assaypilot plan \
  --campaign examples/simulated-enzyme/campaign.json \
  --measurements examples/simulated-enzyme/measurements.csv \
  --candidates examples/simulated-enzyme/candidates.csv \
  --representation physchem --rank-variability 20 -o example_plan
```

`measurements.csv` uses full sequences and `candidates.csv` uses mutation notation, so between
them the example shows both accepted input styles.

### 2. Run it on your own campaign

```bash
# Define what you are measuring.
assaypilot init-campaign \
  --id bglb-thermostability \
  --reference-fasta reference.fasta \
  --assay-name "residual activity after 50C, 30 min" \
  --units "fraction of unheated control" \
  --objective maximize \
  --condition "50C_30min_lysate" \
  --threshold 0.6 \
  --batch-size 24 \
  -o campaign.json

# Check the inputs before trusting anything downstream.
assaypilot validate \
  --campaign campaign.json \
  --measurements measurements.csv \
  --candidates candidates.csv

# Compare methods without committing to a batch.
assaypilot evaluate \
  --campaign campaign.json \
  --measurements measurements.csv \
  --candidates candidates.csv

# Produce the batch and its evidence.
assaypilot plan \
  --campaign campaign.json \
  --measurements measurements.csv \
  --candidates candidates.csv \
  --batch-size 24 \
  --rank-variability 20 \
  -o round4_plan \
  --store campaigns.db
```

### 3. When the results come back

```bash
assaypilot score-round \
  --manifest round4_plan/run_manifest.json \
  --measurements round4_results.csv \
  --previous measurements.csv \
  --round-id R4 \
  --store campaigns.db
```

This scores what was predicted against what was measured, *then* appends the round as a new
campaign version. The earlier plan is left exactly as it was.

### 4. In the browser

```bash
assaypilot app
```

Four screens: import and review, model evidence, batch selection, round history.

---

## What a run actually says

This is real, unedited output from the bundled example — including the parts most tools would
rather not print:

```
24 of 24 requested candidates selected using Simple baseline (one-hot substitutions, ridge)
with the top_ranked policy — supported by held-out evaluation.

Prediction task
---------------
trained on round(s) R1, evaluated on round R2
trained on round(s) R1, R2, evaluated on round R3
(extrapolation evaluation)

Method comparison
-----------------
                                      method_name  spearman  hit_rate_at_k  best_at_k
   Simple baseline (one-hot substitutions, ridge)  0.310995       0.166667   2.435122
Assay-trained model (physchem embeddings + ridge)  0.256996       0.166667   2.811314

Why this method
---------------
assay_trained_embedding_ridge did not improve on baseline_onehot_ridge: Hit rate in the top
batch: +0.002 versus baseline_onehot_ridge (95% interval -0.104 to +0.146), not distinguishable
from the reference within the resampling interval.
No method was distinguishable from the simplest supported method within the resampling
interval, so the simpler method is used.

Warnings
--------
- pretrained_esm_masked_marginal was not available: torch and transformers are not installed...
- 42 of 300 candidates are further from the measured variants than anything the evaluation
  covered (more than 3 substitutions away). The reported performance does not extend to them;
  you can restrict selection to the evaluated range.
- 94% of candidates carry at least one substitution that never appears in the measurements.
  The simple baseline cannot represent those substitutions at all, and the other methods are
  extrapolating.
```

Three things to notice, because they are the design working rather than failing:

- the fancier method had a higher `best_at_k`, and was **still not adopted**, because the
  interval for its improvement on the primary metric spans zero;
- a method that could not run is named, with the reason, instead of disappearing;
- the candidates the evidence does not cover are counted before you spend wells on them.

---

## Input formats

### Measurements (CSV, TSV, or Parquet)

| Column | Required | Notes |
| --- | --- | --- |
| `variant_id` | yes | Unique per distinct sequence. Reused ids with different sequences are rejected. |
| `sequence` | yes\* | Full-length, same length as the reference, standard amino acids only. |
| `mutations` | yes\* | Alternative to `sequence`: `A12G:L30P` or `WT`. Checked against the reference — a stated wild-type residue that does not match is an error, never silently renumbered. |
| `value` | yes | The measured number. `<0.01` and `>5` are read as censored values. |
| `condition_id` | recommended | Rows from another condition are excluded, not pooled. |
| `round_id` | recommended | Unlocks round-based evaluation, which is the most realistic protocol. |
| `replicate_id`, `plate_id` | optional | Replicates always stay in the same evaluation partition. |

\* one of `sequence` or `mutations` is required.

```csv
variant_id,sequence,value,condition_id,round_id,replicate_id
var0001,MKAILVVLL...,1.84,50C_30min_lysate,R1,rep1
var0001,MKAILVVLL...,1.79,50C_30min_lysate,R1,rep2
var0002,MKAILVVLL...,0.42,50C_30min_lysate,R1,rep1
```

### Candidates (CSV, TSV, Parquet, or FASTA)

`candidate_id` plus `sequence` (or `mutations`). Sequences must be the reference length in
version one. Duplicates inside the pool are excluded so a batch never spends two wells on the
same construct.

### Budget

Candidate wells, replicates, and control wells are tracked separately and reported separately. A
batch of 24 candidates with 2 replicates and 4 controls is 52 wells, and the export says so.

---

## How the decision is made

### Step 1 — Choose an evaluation protocol that matches your next step

In preference order:

1. **Round split** — train on earlier rounds, test on a later one. The closest analogue of the
   real task; used whenever usable round labels exist.
2. **Mutation-distance split** — hold out the measured variants that are as far from the
   reference as your candidate pool is.
3. **Mutation-pattern split** — hold out whole sets of positions, testing prediction at
   positions never seen in training.
4. **Random grouped split** — an interpolation diagnostic, labelled as such.

Identical sequences and all their replicates always stay in the same partition.

### Step 2 — Compare the same three methods on the same data

| Method | Implementation | What it is there for |
| --- | --- | --- |
| Simple baseline | One-hot substitution indicators + ridge | Establishes whether protein-model representations add anything |
| Pretrained score | ESM-2 masked marginals; single-site scores summed for combinations | Tests the model without fitting to your labels |
| Assay-trained | Frozen ESM-2 embeddings + scaler/PCA/ridge, all fitted inside training folds | Learns the representation-to-outcome relationship for your assay |

The baseline is tuned as carefully as the other two: a deliberately weak baseline would make
everything else look good. The pretrained score's additive combination of single-site terms
cannot represent epistasis, and every multi-substitution score says so.

### Step 3 — Apply one stated rule

> Start from the simplest available method. Adopt a more complex method only when the paired
> bootstrap 95% interval for its improvement on the primary metric lies entirely above zero.
> Otherwise keep the simpler method.

The bootstrap resamples **distinct held-out sequences** (replicates move together) and is
**paired**, so shared resampling noise does not inflate differences. Metrics follow the decision:
rank correlation for ordering, hit rate in the top batch when a success threshold was defined in
advance, and best measured value at the selection budget — each shown next to what random
selection of the same size would have achieved.

The result is labelled `supported` only when the split was an extrapolation protocol, there were
enough distinct held-out sequences, and the chosen method ordered them better than chance.
Otherwise it is labelled `exploratory`, in the report, in the exports, and in the manifest.

### Step 4 — Check that the evidence covers your candidates

For each candidate: distance to the nearest measured variant, how many of its substitutions have
never been measured, and whether it falls inside the region the evaluation actually covered.
Candidates beyond that region are flagged, and `--restrict-to-evaluated` excludes them entirely.

### Step 5 — Filter, then rank, then select

Filtering is deterministic and happens *before* scoring: invalid sequences, duplicates,
already-measured sequences (unless you ask for repeats), substitutions at locked positions, and
candidates over the substitution limit. Then one of two policies:

- **`top_ranked`** (default) — the highest-ranked eligible candidates.
- **`diversity`** — after each pick, suppress candidates closer than a visible Hamming-distance
  threshold. If the batch cannot be filled, the shortfall is reported; the constraint is never
  quietly relaxed.

Version one does **not** select by "highest uncertainty". Uncertainty estimates for protein
models are not reliably calibrated, and uncertainty-based selection does not consistently beat
simpler approaches [3]. Bootstrap refits are reported as *rank variability* — how much the
ranking depends on which measurements you happen to have — and never as a probability of success.

### When the evidence is weak

| Situation | What AssayPilot does |
| --- | --- |
| No measured data | Allows data preparation and an exploratory pretrained ranking; claims no assay-specific validation |
| Too little data, or unstable comparisons | Reports exploratory results and offers a diversity-based batch; claims no reliable ordering of methods |
| No supported gain over the baseline | Uses the simpler method and says why |
| Candidates beyond the evaluated patterns | Flags the unsupported region; offers to restrict selection to the evaluated range |
| Conditions changed between rounds | Treats it as a separate assay context; does not pool measurements |
| Conflicting replicates, censored values | Exposes them, keeps raw values; censored values are excluded from regression rather than treated as exact |

---

## What comes out

`assaypilot plan -o round4_plan` writes:

| File | Contents |
| --- | --- |
| `selected_variants.csv` | The batch: immutable `design_id`, rank, sequence, mutations, score, predicted value, observed value where it exists, distance to measured variants, coverage flag, ranking stability, and the reason it was selected |
| `selected_variants.fasta` | The same sequences, keyed by `design_id` |
| `all_predictions.csv` | Every candidate, eligible or not, with its score and eligibility |
| `exclusions.csv` | Every row set aside, with a code and a reason |
| `candidate_coverage.csv` | Per-candidate distance and novelty |
| `method_comparison.csv` | Per-split and aggregate metrics for every method |
| `report.html` | Self-contained report: the batch, the evidence, the limits, the exclusions, how to reproduce it |
| `run_manifest.json` | Input checksums, transformations, partition assignments, seeds, chosen hyperparameters, pinned checkpoint revision, package versions, every prediction, every exclusion, the selected batch |
| `campaign.json` | The campaign definition used |

Three categories are kept strictly apart in every file: `observed_value` was measured,
`score`/`predicted_value` came from a model, and `selection_reason` says why a candidate is in
the batch. `predicted_value` is empty for the pretrained score, because that method produces a
score on an arbitrary scale, not a predicted assay value.

`design_id` is derived from the campaign and the sequence, so the same construct always receives
the same identifier — that is what makes next round's results matchable back to this plan.

---

## The round loop

```
measurements ──► validate ──► evaluate ──► select ──► export
      ▲                                                  │
      │                                                  ▼
      └──── score-round ◄──── new results ◄──── run the experiment
```

`score-round` is the only prospective evidence the tool ever produces: predictions made before
the measurements existed, compared with the measurements that followed. Everything else is
retrospective. Each round creates a new campaign version; earlier predictions, selections, and
reports are never modified.

---

## Command reference

| Command | Purpose |
| --- | --- |
| `assaypilot init-campaign` | Create `campaign.json`: reference, assay name, units, objective, condition, threshold |
| `assaypilot validate` | Check inputs, show exclusions and the evaluation protocol that would be used |
| `assaypilot evaluate` | Compare methods and print the recommendation without selecting a batch |
| `assaypilot plan` | Full workflow: evaluate, select, export, optionally record in a campaign store |
| `assaypilot score-round` | Score a plan against the measurements it produced, then append the round |
| `assaypilot demo` | Generate simulated data and run everything on it |
| `assaypilot benchmark` | Replay a measured dataset round by round and compare strategies |
| `assaypilot app` | Launch the Streamlit interface |

Useful flags for `plan`: `--policy diversity --min-distance 2`, `--lock-position 145`,
`--max-substitutions 4`, `--restrict-to-evaluated`, `--split round`, `--representation physchem`,
`--esm-revision <commit>`, `--rank-variability 20`, `--seed 0`, `--store campaigns.db`.

Run `assaypilot <command> --help` for the full list.

---

## Python API

```python
from assaypilot import Campaign, load_candidates, load_measurements, plan_batch
from assaypilot.models import SuiteConfig
from assaypilot.reporting import export_plan

campaign = Campaign.from_json("campaign.json")
measurements = load_measurements("measurements.csv", campaign, strict=False)
candidates = load_candidates("candidates.csv", campaign, strict=False)

print("\n".join(measurements.report.summary_lines()))

plan = plan_batch(
    campaign,
    candidates,
    measurements,
    suite=SuiteConfig(representation="esm2", esm_revision="<commit>"),
    policy="diversity",
    rank_variability_refits=20,
    seed=0,
)

print(plan.headline())
print(plan.evaluation.recommendation.reasons)
export_plan(plan, "round4_plan")
```

The modelling and evaluation code is shared by the CLI, the interface, and the benchmark runner,
so there is exactly one implementation of the decision to audit.

---

## Scope of version one

**In scope**

- One protein, one quantitative assay, one condition per campaign
- Substitution-only variants of the reference length, standard amino acids
- A user-supplied candidate pool (no de novo generation)
- Three methods, two selection policies, versioned rounds
- Exports that work without AssayPilot installed

**Design envelope** (subject to your hardware): up to ~500 measured variants and ~5,000
candidates of 50–500 residues. Runtime depends on sequence length, candidate count, hardware,
and checkpoint size; `assaypilot benchmark` records wall-clock time so you can measure it on your
own machine rather than trusting a promise of real-time operation.

**Not in version one**: insertions and deletions, multi-objective optimization, structural
prediction, automatic normalization across incompatible assay batches, foundation-model
fine-tuning, de novo sequence design.

---

## What AssayPilot claims, and what it does not

**It claims**: that it predicts the recorded assay outcome under the recorded condition, with
evidence from held-out measurements of that campaign; that its selections are deterministic,
constrained, and fully recorded; and that a run can be reproduced from its manifest.

**It does not claim**: a numerical improvement in hit rate — that is an outcome to measure, not a
promised percentage; that uncertainty estimates are calibrated — no interval is presented as
calibrated unless held-out coverage has been assessed, which version one does not do; that a
pretrained model has never seen your sequences — published datasets may overlap pretraining
corpora; or novelty relative to existing systems.

`docs/claims.md` states this in full, including how each limitation is enforced in code.

---

## How it relates to existing tools

[EVOLVEpro](https://github.com/mat10d/EvolvePro) combines protein-language-model embeddings with
experimental measurements across rounds. [OpenProtein](https://docs.openprotein.ai/opmodels/)
offers assay-trained models, uncertainty, and constrained design.
[ProteusAI](https://github.com/jonfunk21/ProteusAI) is an open-source protein engineering library
with a web application. These systems already cover much of this category, and AssayPilot makes
no claim that they lack evaluation.

What AssayPilot aims to offer is a narrow, complete, inspectable chain: a declared next-round
task → an evaluation that matches it → a documented selection rule → preserved results when the
next round returns. Local execution, a small fixed comparison suite, and explicit handling of
unsupported cases are what make that chain auditable. Inputs and exports are deliberately
portable, so it can be used as a standalone tool or as a module inside another platform.

---

## Documentation

| Page | What it covers |
| --- | --- |
| [docs/claims.md](docs/claims.md) | Every claim the software makes, what it refuses to claim, and the code and test enforcing each one |
| [docs/methodology.md](docs/methodology.md) | Why the splits, metrics, bootstrap, and selection rule are what they are |
| [docs/data-formats.md](docs/data-formats.md) | Every input column and output file, field by field |
| [docs/representations.md](docs/representations.md) | One-hot, ESM-2, physicochemical features, the embedding cache, adding your own |
| [docs/benchmarks.md](docs/benchmarks.md) | Replaying public datasets, and the reporting rules that keep it honest |
| [docs/testing.md](docs/testing.md) | What each test module protects against |

---

## Project layout

```
src/assaypilot/
  campaign.py          Campaign and selection settings
  sequences.py         Validation, mutation notation, distances
  data/                Loaders, FASTA, validation reports and exclusion codes
  features/            Representations: ESM-2, physicochemical, one-hot; embedding cache
  models/              Baseline, pretrained score, assay-trained model, suite registry
  evaluation/          Splits, metrics, paired comparison, coverage, rank variability
  selection/           Eligibility filters and selection policies
  reporting/           Run manifest, exports, HTML report
  benchmarks/          Retrospective replay harness
  app/                 Streamlit interface
  pipeline.py          The one implementation of the decision
  store.py             SQLite campaign history (append-only)
  scoring.py           Scoring a plan against the round that followed
  cli.py               Command line
docs/                  Claims, methodology, data formats, benchmarks, representations, testing
examples/              A labelled synthetic campaign you can run immediately
scripts/               Fetch public datasets, run replays, regenerate the example
tests/                 Validation, leakage, constraints, determinism, end-to-end
```

Start reading at [`pipeline.py`](src/assaypilot/pipeline.py): it is the single implementation of
the decision, and every other entry point calls it.

---

## Development

```bash
pip install -e ".[dev]"
pytest                      # fast suite: no model downloads, no network
pytest -m slow              # adds the ESM-2 tests (downloads an 8M checkpoint)
ruff check src tests scripts
ruff format src tests scripts
```

The fast suite is verified to run with the network disabled and an empty Hugging Face cache,
even when torch is installed. The ESM-2 paths — tokenizer offsets, padding-aware pooling,
additivity of masked-marginal scores, revision recording, cache namespacing — are covered by
`tests/test_esm.py`, which is opt-in via `-m slow` and defaults to the smallest ESM-2
checkpoint. Point it at another with
`ASSAYPILOT_TEST_CHECKPOINT=facebook/esm2_t12_35M_UR50D pytest -m slow`.

The test suite deliberately covers the failure cases that matter in this domain: sequence
validation, replicate leakage, preprocessing leakage, constraint enforcement, round matching, and
deterministic reruns. `docs/testing.md` explains what each group protects against.

Contributions are welcome — see `CONTRIBUTING.md`.

---

## Status

Version 0.1.0, alpha. The workflow is complete end to end: validation, evaluation, selection,
exports, round scoring, campaign history, CLI, interface, and the replay harness all work and are
covered by tests.

Public-data results exist: twelve ProteinGym assays in three runs — five used while building
the harness, six drawn at random by a rule fixed beforehand, and GFP as a deliberate case study.
Each campaign gets 168 wells (48 initial, then 5 rounds of 24) over 5 seeds.

On the six **held-out** assays the assay-trained model found 31.9 hits against 16.5 for random
selection, with the pretrained ESM-2 score at 29.1 and the one-hot baseline at 19.8. But the
useful finding is in the third run: **the ranking of methods reverses between datasets.** On
GFP — 51,714 variants, 98% of them multi-substitution — the pretrained score came *last*, below
random, losing on every seed, because it sums single-site terms and cannot represent epistasis;
the one-hot baseline, worst on the single-mutant assays, came first by a factor of 2.7 over
random. Every method has at least one assay where it does no better than chance.

That is the case for evaluating per campaign rather than picking a method once, globally — and
it is what AssayPilot does. Full tables, base-rate caveats, and the commands are in
[docs/benchmarks.md](docs/benchmarks.md#measured-results).

What is **not** established: a comparison against other tools (EVOLVEpro and similar are not in
these tables), statistical significance from five seeds, or anything about assays beyond these
twelve.

---

## References

1. **ProteinGym** — public benchmarks for mutation-effect prediction across assays and prediction
   settings. <https://proteingym.org/>
2. **FLIP** — protein-engineering evaluation tasks with biologically motivated train/test
   partitions. <https://github.com/J-SNACKKB/FLIP>
3. Greenman, Amini & Yang (2025). *Benchmarking uncertainty quantification for protein
   engineering.* PLOS Computational Biology.
   <https://doi.org/10.1371/journal.pcbi.1012639>
4. **ESM-2 model card** — checkpoint sizes and loading guidance; the 35M checkpoint is the
   default here. <https://huggingface.co/facebook/esm2_t12_35M_UR50D>
5. **EVOLVEpro** — <https://github.com/mat10d/EvolvePro>
6. **OpenProtein** — <https://docs.openprotein.ai/opmodels/>
7. **ProteusAI** — <https://github.com/jonfunk21/ProteusAI>

These sources establish that the underlying methods are available, that the workflow category is
useful enough to have substantial prior work, and that generalization and uncertainty require
careful evaluation. They do not establish a numerical benefit for this implementation.

---

## License and citation

Apache-2.0. See [LICENSE](LICENSE).

If AssayPilot contributes to published work, please cite it using [CITATION.cff](CITATION.cff),
and cite the underlying methods and datasets you used — ESM-2, ProteinGym, FLIP — directly.
