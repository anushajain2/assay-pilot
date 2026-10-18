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

## A first run: five ProteinGym assays

Produced by the protocol above on 19 October 2026. These are the only measured
results this repository reports, and they come with the limits listed under
"What a replay cannot tell you".

**Setup.** ProteinGym v1.1 substitution assays, five of them chosen as enzymes
with single-substitution coverage and workable pool sizes. 48 initial
observations revealed at random, then 5 rounds of 24 — 168 wells in total —
repeated over 5 seeds, with every strategy sharing the same initial set at each
seed. Representation: `facebook/esm2_t12_35M_UR50D` pinned at revision
`6fbf070e65b0b7291e7bbcd451118c216cff79d8`. A "hit" is the top decile of each
assay's own measured values, a threshold fixed before the replay.

| Assay | Variants | Reference |
| --- | --- | --- |
| AMIE_PSEAE_Wrenbeck_2017 (amidase) | 6,227 | 346 aa |
| BLAT_ECOLX_Jacquier_2013 (β-lactamase) | 989 | 286 aa |
| BLAT_ECOLX_Stiffler_2015 (β-lactamase) | 4,996 | 286 aa |
| DYR_ECOLI_Thompson_2019 (DHFR) | 2,363 | 159 aa |
| ESTA_BACSU_Nutschel_2020 (esterase) | 2,172 | 212 aa |

### Hits found in 168 wells, mean of 5 seeds

| Assay | random | baseline | ESM-2 score | assay-trained | assay-trained +div |
| --- | ---: | ---: | ---: | ---: | ---: |
| AMIE_PSEAE_Wrenbeck_2017 | 16.8 | 12.2 | **28.4** | 26.2 | 26.2 |
| BLAT_ECOLX_Jacquier_2013 | 63.4 | 65.8 | **87.0** | 81.0 | 81.6 |
| BLAT_ECOLX_Stiffler_2015 | 16.6 | 19.0 | 26.0 | **31.4** | 28.4 |
| DYR_ECOLI_Thompson_2019 | 18.8 | 11.8 | **20.0** | 19.6 | 18.6 |
| ESTA_BACSU_Nutschel_2020 | 19.6 | 21.0 | 42.6 | **46.2** | 34.8 |
| **mean of per-assay means** | 27.0 | 26.0 | **40.8** | **40.9** | 37.9 |

### What that does and does not show

- **The two ESM-2 methods find about 1.5× as many hits as random**, 40.8 and
  40.9 against 27.0, averaged over assays.
- **The one-hot baseline does not**: 26.0, slightly *below* random, and clearly
  below it on two assays. With 48 initial measurements on pools of thousands,
  most candidates carry substitutions the baseline has never seen, so it
  predicts near the training mean for them. This is the case the baseline
  exists to reveal, and here it says the protein-model representation is doing
  real work.
- **One assay shows nothing.** On DYR_ECOLI_Thompson_2019 every strategy lands
  between 11.8 and 20.0 hits against 18.8 for random: no method beat chance,
  and the baseline did worse than it. It is reported here rather than dropped.
- **Diversity is not free.** The diversity variant found fewer hits than plain
  ranking on four of five assays (37.9 against 40.9 overall). On this evidence
  it is a constraint to reach for when you want spread, not a default.
- **Five seeds is not many.** Counted per seed, the best-value-found metric
  beat random in 1–5 seeds out of 5 depending on assay and method; the
  differences in that metric are mostly within noise even where the hit counts
  separate cleanly.

### A note on the metric that did not work

"Best value found" saturates at this budget. Expressed as a percentile of each
assay's own pool, every strategy — random included — lands between the 96th and
100th percentile, because 168 draws from a few thousand variants will find
something near the top whatever you do. The aggregate therefore reports hit
counts and within-assay percentiles only; raw assay values are never averaged
across assays, since an esterase activity near 50 and a fitness near 1 do not
belong in the same mean.

### Reproducing it

```bash
python scripts/fetch_proteingym.py --out data/proteingym \
  --assay AMIE_PSEAE_Wrenbeck_2017 --assay BLAT_ECOLX_Jacquier_2013 \
  --assay BLAT_ECOLX_Stiffler_2015 --assay DYR_ECOLI_Thompson_2019 \
  --assay ESTA_BACSU_Nutschel_2020

python scripts/run_benchmarks.py --data data/proteingym/campaigns --out results/ \
  --n-initial 48 --batch-size 24 --rounds 5 --seeds 0 1 2 3 4 \
  --representation esm2 --esm-revision 6fbf070e65b0b7291e7bbcd451118c216cff79d8
```

Runtime on an M-series Mac, with embeddings cached between assays: about 25
minutes for all five. The masked-marginal score costs one forward pass per
mutated position of the reference (~3s per selection against ~0.4s for the
embedding model), paid once per assay and then cached.

These five assays were also the development set for the harness itself, so
treat them as a demonstration that the pipeline produces sane numbers, not as
an independent evaluation. A reporting run should use assays that were not
used while building it.

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
