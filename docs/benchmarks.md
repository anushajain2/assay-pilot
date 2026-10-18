# Benchmarking the whole decision process

Evaluating a *model* and evaluating a *decision process* are different
exercises. AssayPilot's benchmark harness does the second: it replays a fully
measured dataset round by round and compares strategies on the thing that
matters — what the campaign would have found within its budget.

**This repository ships no benchmark results.** The protocol below is what a
release should run; the numbers have to be produced on datasets you download,
with the versions recorded.

---

## The replay protocol

1. Take a dataset where every variant's outcome is known.
2. Reveal a random initial set (`--n-initial`).
3. Fit the strategy on what is revealed; select a batch of `--batch-size`.
4. Reveal **only that batch's** outcomes.
5. Repeat for `--rounds` rounds.
6. Repeat the whole thing for each seed, with the *same* initial set shared by
   every strategy at that seed.

```bash
assaypilot benchmark \
  --campaign campaign.json \
  --measurements public_assay.csv \
  --n-initial 48 --batch-size 24 --rounds 5 \
  --seeds 0 1 2 3 4 \
  --representation esm2 --esm-revision <commit> \
  -o benchmark_output
```

### Strategies compared

| Strategy | What it tests |
| --- | --- |
| `random` | The floor. Any strategy that cannot beat it is not helping. |
| `baseline_onehot_ridge` | Does a linear model over substitutions suffice? |
| `pretrained_esm_masked_marginal` | How far do you get with no labels at all? |
| `assay_trained_embedding_ridge` | Does learning this assay's mapping help? |
| `assay_trained_embedding_ridge+diversity` | Does spreading the batch out help, or not? |

An established active-learning implementation such as EVOLVEpro should be added
to this table wherever its assumptions fit the dataset; matching candidate
pools, initial observations, batch budgets, and seeds is the point of the
harness.

### What is reported

Per round and seed: batch best, batch mean, best found so far, **best among
the variants the strategy itself selected**, cumulative discoveries above the
predeclared threshold, cumulative experiments, and wall-clock seconds.

`best_selected` is the one to compare on. `best_found` includes the shared
initial reveal, so when that set already contains the pool's best variant it
saturates and every strategy looks identical — which says nothing about the
strategies. `compare_strategies` defaults to `best_selected` and pairs
strategies **by seed**, because the seed fixes the initial set and an unpaired
comparison would mostly measure which seed got lucky.

### What a replay cannot tell you

Both limits are printed with every result:

- selection is restricted to variants whose outcome the dataset already
  contains, so the candidate pool is not the pool a real campaign faces;
- the exercise is retrospective: it shows what a strategy *would* have chosen,
  not what a new experiment would have produced.

---

## Choosing datasets

**ProteinGym** (<https://proteingym.org/>) — mutation-effect benchmarks across
many assays, with defined prediction settings.

**FLIP** (<https://github.com/J-SNACKKB/FLIP>) — protein-engineering tasks with
biologically motivated train/test partitions. When a FLIP split is used, keep
the published assignment (`assaypilot.datasets.load_flip_split`) so results stay
comparable with the published task.

Start with fluorescent-protein and enzyme assays whose readouts and variant
coverage fit version one's scope: substitution-only variants of a single
reference, one quantitative readout.

```python
from assaypilot.datasets import load_proteingym_assay

campaign, measurements = load_proteingym_assay(
    "DMS_ProteinGym_substitutions/SOME_ASSAY.csv",
    assay_name="DMS score (published)",
    assay_units="score (dataset units)",
)
```

### Rules for reporting

- **Separate development assays from reporting assays.** Datasets used while
  building and debugging must not be the ones a headline number is quoted
  from.
- **Report per-assay results as well as aggregates**, so one large assay cannot
  dominate a mean.
- **Keep the failures in.** Assays where no method beat the baseline, and
  settings that did not work, belong in the report. A benchmark table with no
  losses is a selection artifact.
- **Do not claim unseen sequences.** Published assay sequences may overlap
  protein-language-model pretraining corpora. No result here may be described
  as evaluation on sequences the model has never seen.
- **Record the dataset release and the checkpoint revision** next to every
  number.

---

## Reproducing a published figure

Anything published from this repository should be regenerable by a script in
`scripts/`, taking a data directory and an output directory, with seeds fixed
and the environment recorded. The pattern:

```bash
python scripts/fetch_proteingym.py --out data/proteingym        # you download
python scripts/run_benchmarks.py --data data/proteingym --out results/
```

`scripts/run_benchmarks.py` writes `replay_rounds.csv`,
`replay_summary.csv`, and `replay_config.json` per assay, plus a combined
table. Every figure in a report should cite the CSV it came from.

---

## Measuring cost, not promising speed

Runtime depends on sequence length, candidate pool size, hardware, and
checkpoint size. The harness records wall-clock seconds per selection step so
you can measure your own limits. Publish measured numbers with the hardware
they were measured on; do not promise real-time operation.

`scripts/profile_runtime.py` measures the numbers worth publishing — embedding
throughput, planning time across pool sizes with a cold and a warm cache, and
peak memory — and writes them with the hardware and package versions beside
them:

```bash
python scripts/profile_runtime.py --out results/profile \
  --representation esm2 --esm-revision <commit> \
  --pool-sizes 100 500 1000 5000
```

It writes `profile.csv`, `profile.md` (a table ready to paste into a report),
and `environment.json`. The table says in its own header that it is a
measurement from one machine, not a performance claim.
