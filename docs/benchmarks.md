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

## Measured results

Three runs, in the order they were done. Together they are the only measured
results this repository reports, and the third one contradicts the first —
which is the most useful thing in this section.

Common protocol: 48 initial observations revealed at random, then 5 rounds of
24, so 168 wells per campaign; 5 seeds, with every strategy sharing the same
initial set at each seed; representation `facebook/esm2_t12_35M_UR50D` pinned
at revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`; ProteinGym v1.1. A
"hit" is a variant at or above the top decile of that assay's own measured
values, fixed before the replay.

### How to read the hit counts

A hit count only means something next to the **base rate** — the share of the
whole pool that counts as a hit. Where many variants tie at a floor or ceiling
value, the top-decile threshold can cover most of the pool, random selection
then scores highly, and the metric separates nothing. Two assays here are like
that (CcdB at 79%, one β-lactamase study at 38%); they are reported but
excluded from the aggregate hit column, and `run_benchmarks.py` now prints the
base rate and flags them automatically.

With a 10% base rate, random selection finds about 17 hits in 168 wells.

### 1. Development set (five enzyme assays)

These five were used while building the harness, so they are a demonstration
that the pipeline produces sane numbers, not an evaluation.

| Assay | base rate | random | baseline | ESM-2 score | assay-trained | +diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMIE_PSEAE_Wrenbeck_2017 | 10% | 16.8 | 12.2 | **28.4** | 26.2 | 26.2 |
| BLAT_ECOLX_Jacquier_2013 | 38% ⚠ | 63.4 | 65.8 | 87.0 | 81.0 | 81.6 |
| BLAT_ECOLX_Stiffler_2015 | 10% | 16.6 | 19.0 | 26.0 | **31.4** | 28.4 |
| DYR_ECOLI_Thompson_2019 | 10% | 18.8 | 11.8 | **20.0** | 19.6 | 18.6 |
| ESTA_BACSU_Nutschel_2020 | 10% | 19.6 | 21.0 | 42.6 | **46.2** | 34.8 |
| **mean, usable thresholds** | | 18.0 | 16.0 | **29.3** | **30.9** | 27.0 |

The one-hot baseline came out *below* random. With 48 initial measurements on
pools of thousands of single mutants, most candidates carry a substitution the
baseline has never seen, so it predicts near the training mean for them. On
DYR nothing beat random at all.

### 2. Held-out set (six assays, chosen before the run)

Selected by a stated rule, not by eye: from the 119 ProteinGym substitution
assays with 1,000–8,000 variants and a reference under 450 residues, excluding
the five development assays, six were drawn with `numpy.default_rng(20261019)`.
The protocol was written down before any of them ran.

| Assay | base rate | random | baseline | ESM-2 score | assay-trained | +diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CCDB_ECOLI_Tripathi_2016 | 79% ⚠ | 131.8 | 141.0 | 126.8 | 145.4 | 149.6 |
| FKBP3_HUMAN_Tsuboyama_2023_2KFV | 10% | 16.4 | 15.2 | 11.4 | **28.4** | 28.4 |
| KCNE1_HUMAN_Muhammad_2023_function | 10% | 15.6 | **33.8** | 31.6 | 15.6 | 17.0 |
| NUSG_MYCTU_Tsuboyama_2023_2MI6 | 10% | 15.8 | 9.6 | 37.8 | **70.2** | 57.8 |
| PTEN_HUMAN_Mighell_2018 | 10% | 16.6 | 16.2 | **31.4** | 21.4 | 21.4 |
| RL40A_YEAST_Roscoe_2014 | 10% | 18.2 | 24.4 | **33.2** | 23.8 | 23.6 |
| **mean, usable thresholds** | | 16.5 | 19.8 | 29.1 | **31.9** | 29.6 |

The ordering survives the move to held-out data — assay-trained 31.9 and
ESM-2 score 29.1 against 16.5 for random — but every method has at least one
assay where it does no better than chance. The baseline loses to random on
three of six, the pretrained score on two, and the assay-trained model ties it
on KCNE1 while winning four-fold on NusG. No method wins everywhere.

### 3. GFP: where the additive approximation breaks

`GFP_AEQVI_Sarkisyan_2016`, 51,714 variants of which 50,630 carry **more than
one substitution**. Every assay above is single-substitution only, so none of
them tested the pretrained score's documented limit: it sums single-site
log-ratios, which assumes substitutions act independently. On single mutants
there is nothing to approximate. Here there is.

| Strategy | hits in 168 wells | vs random, best value (5 seeds) |
| --- | ---: | --- |
| baseline (one-hot) | **51.6** | +0.034, won 3/5 seeds |
| assay-trained +diversity | 23.2 | −0.017, won 2/5 |
| assay-trained | 21.4 | −0.007, won 3/5 |
| random | 18.8 | — |
| ESM-2 score | **10.6** | −0.196, won **0/5** |

**The pretrained score finished below random**, and lost on every seed. That is
the additive approximation failing exactly where the method's own caveat says
it should. Meanwhile the one-hot baseline — worst on the single-mutant assays —
is the best method here by a factor of 2.7 over random, because 51,714 variants
over a limited set of positions means it has seen nearly every substitution
before.

### What these runs support

The headline is not that one method wins. It is that **the ranking of methods
reverses between datasets**: the baseline was the worst method on the
development assays and the best on GFP; the pretrained score was the best on
several single-mutant assays and worse than random on GFP. A tool that picked a
method once, globally, would be wrong half the time here.

That is the case for evaluating per campaign, on the campaign's own held-out
measurements, and for preferring the simpler method unless the evidence clears
a stated bar — which is what AssayPilot does.

### What they do not support

- No claim about any assay not listed. Six held-out assays is a small sample,
  and the spread between them is wide.
- No comparison against other tools. EVOLVEpro and similar systems are not in
  this table; adding one is the obvious next step.
- No statistical significance claim. Five seeds, paired by initial set; most
  per-assay differences in *best value found* sit inside that noise even where
  hit counts separate cleanly.
- Nothing about unseen sequences. These assays may overlap ESM-2's pretraining
  data, so none of this measures generalisation to genuinely novel proteins.
- Retrospective throughout: selection is restricted to variants the dataset
  already measured.

### Reproducing

```bash
python scripts/fetch_proteingym.py --out data/proteingym --list     # 217 assays
python scripts/fetch_proteingym.py --out data/proteingym --assay NUSG_MYCTU_Tsuboyama_2023_2MI6
python scripts/run_benchmarks.py --data data/proteingym/campaigns --out results/ \
  --n-initial 48 --batch-size 24 --rounds 5 --seeds 0 1 2 3 4 \
  --representation esm2 --esm-revision 6fbf070e65b0b7291e7bbcd451118c216cff79d8
```

On an M-series Mac with embeddings cached: about 25 minutes for five
single-mutant assays, about 20 for GFP once its 51,714 embeddings are computed
(~9 minutes at ~100 sequences/s). The masked-marginal score costs one forward
pass per mutated position of the reference, paid once per assay.

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
