# Methodology

How AssayPilot turns measurements into a batch, and why each step is the way it
is. For the strict list of what is and is not asserted, see
[claims.md](claims.md).

---

## The chain

```
campaign  →  validated data  →  evaluation protocol  →  method comparison
          →  selection rule  →  refit on everything  →  coverage check
          →  eligibility filter  →  ranking  →  policy  →  plan + manifest
```

Each arrow is a decision that is recorded. The point of the design is that
someone who disagrees with a decision can see it, change it, and rerun.

---

## 1. The campaign is the unit of claim

A campaign fixes a reference sequence, an assay name, units, a direction of
improvement, and an experimental condition. Everything downstream inherits
them.

This is not bookkeeping. The reason a campaign cannot span two conditions is
that the model's claim would stop being checkable: "predicts activity" is not a
testable statement, while "predicts initial rate on pNPA at 37 °C, pH 7.4, in
lysate" is. When conditions change between rounds, the honest move is a new
campaign, and the loader enforces it by excluding foreign-condition rows with
that reason attached.

**Success thresholds must be declared in advance.** A hit rate computed against
a threshold chosen after seeing results is not a measurement, it is a
description. `Campaign.is_hit` raises if no threshold was set, and
`choose_primary_metric` falls back to rank correlation.

---

## 2. Validation: reject, never repair

Three classes of input problem get three different treatments:

| Problem | Treatment |
| --- | --- |
| The file is wrong (invalid residues, length mismatch, mutation notation contradicting the reference, duplicate ids) | **Blocking.** Fix the file, or continue explicitly with `--skip-invalid`, with every row recorded. |
| The row is not usable here (another condition, empty value, duplicate candidate, already measured) | **Excluded** with a code and a reason, kept in `exclusions.csv`. |
| The data is awkward but real (censored values, disagreeing replicates, outliers) | **Exposed, not resolved.** Raw values stay; censored values are excluded from regression only. |

The hardest rule to hold to is the third. A tool that quietly winsorises
outliers or averages away a disagreeing replicate produces nicer-looking
metrics and a worse experiment.

**Mutation numbering is never adjusted.** If a row says `A12G` and the
reference has `S` at position 12, that is an error, because the alternative —
guessing an offset — silently reinterprets the whole dataset.

---

## 3. The evaluation protocol follows the next step

A random split answers: *can the model interpolate among variants like the ones
it has seen?* That is rarely the question. The next batch usually sits further
out than the training data, so the evaluation has to sit further out too.

Preference order, implemented in `choose_split_plan`:

1. **Round split.** Train on earlier rounds, test on the next one. Forward
   chaining, so each split is a real "what would I have predicted then?"
   question. A sequence remeasured later stays with the training rounds and is
   dropped from the test side.
2. **Mutation-distance split.** Hold out variants with at least *k*
   substitutions, where *k* is aimed at the candidate pool's median. Produced
   only when the holdout is 15–50% of sequences, so it is neither trivial nor
   starving the training set.
3. **Mutation-pattern split.** Hold out whole sets of positions. Variants
   mixing held-out and training positions are dropped from that fold and
   counted, because keeping them would leak.
4. **Random grouped split.** The interpolation diagnostic. Labelled as such
   wherever it appears, and it forces the evidence level to `exploratory`.

Minimum sizes (8 training and 5 evaluation sequences) stop a split from
existing at all when it would be meaningless.

---

## 4. The comparison suite is small and fixed

Three methods, chosen so each answers a question the others cannot:

- **One-hot substitutions + ridge.** Would a linear model over the
  substitutions themselves have done just as well? This is the question a
  protein-model result has to beat. The baseline's penalty is tuned by grouped
  inner CV — a deliberately weak baseline is a way of lying.
- **ESM-2 masked marginals.** How much can be had without labels at all? If
  this ranks held-out variants as well as the trained model, the labels were
  not adding much.
- **Frozen ESM-2 embeddings + ridge head.** Does learning the
  representation-to-outcome mapping for *this* assay help?

Why not more methods? Searching across many models until one wins inflates
apparent performance, especially on the small datasets a real campaign has. The
suite is fixed so the comparison means something.

**Hyperparameters.** Ridge penalty and PCA size are chosen by grouped inner
cross-validation on the training rows of each outer split. PCA sizes are
filtered against the *smallest inner fold*, not the full training set.

---

## 5. Metrics follow the decision

| Metric | Question it answers |
| --- | --- |
| Spearman rank correlation | Does the method order variants at all? |
| Hit rate in the top *k* | Of the *k* it would have picked, how many crossed the predeclared threshold? |
| Best value at *k* | What is the best result the budget would have found? |
| Mean value at *k* | How good is the batch on average? |

All are reported next to a **random reference** — the expectation of picking *k*
at random from the same held-out set — so a positive-looking number can be
compared against doing nothing clever.

The primary metric is fixed **before any model is fitted**, so the choice
cannot follow the results. Hit rate is used only when it can actually separate
methods — a threshold declared in advance, the smallest split at least twice
the batch size, at most 25% of measured variants already clearing the
threshold, and at least five hits held out. A threshold that most measured
variants clear puts random selection near the ceiling too, so the number stops
meaning anything; in that case rank correlation is used and the report says
why. Otherwise rank correlation. The smallest
split governs, because a pooled row count can clear the bar while every
individual fold is almost entirely top-batch — which makes the hit rate nearly
constant by construction. Every split is also scored at the same budget `k`, so
the average across splits is one quantity rather than a mixture of two.

---

## 6. Paired bootstrap over independent units

Comparing two methods by their point estimates ignores that both were scored on
the same small sample. AssayPilot resamples **distinct held-out sequences**
(replicates move with their sequence) and computes both methods' metrics on the
*same* resample, giving a distribution of the *difference*. The 95% percentile
interval of that distribution is what the decision rule reads.

With several splits, the difference is averaged across splits within each
bootstrap replicate, so the interval reflects both sampling noise and
split-to-split variation.

---

## 7. The selection rule

> Start from the simplest available method. Adopt a more complex method only
> when the paired bootstrap 95% interval for its improvement on the primary
> metric lies entirely above zero. Otherwise keep the simpler method.

Consequences, all intended:

- a 0.02 lead with an interval spanning zero changes nothing;
- the baseline wins ties, which is the right default when the alternative costs
  a model download and an inference budget;
- the reasons are recorded as sentences, not just numbers.

**Evidence levels.** `supported` requires an extrapolation split, ≥20 distinct
held-out sequences, and beating chance on whichever metric drove the decision —
zero for a rank correlation, and the random-selection reference for a hit rate. Anything else is
`exploratory`. With no measurements at all, the level is `none` and the ranking
is label-free.

---

## 8. Coverage: does the evidence reach these candidates?

A method can score well and still be unsupported for a specific candidate. For
each candidate AssayPilot computes the distance to the nearest measured
variant, the number of substitutions never seen in any measurement, and whether
it lies within the furthest distance the evaluation covered.

Candidates outside that region are flagged. `--restrict-to-evaluated` excludes
them. Neither is automatic, because sometimes reaching beyond the evidence is
exactly the experiment you want — as long as you know you are doing it.

---

## 9. Filter first, then rank

Eligibility is deterministic and happens before scoring, so a high prediction
can never pull an ineligible sequence into a batch:

1. substitutions at locked positions,
2. more substitutions than the limit,
3. already measured (unless repeats are requested),
4. an optional restriction to the evaluated region.

Then the policy:

- **`top_ranked`** takes the best *k*.
- **`diversity`** suppresses candidates within a visible Hamming distance of
  one already chosen. If the batch cannot be filled, the shortfall is reported
  and the constraint is **not** relaxed.

Diversity is an option to test, not an assumed improvement:
`assaypilot benchmark` runs it against pure ranking so the question can be
answered for your pool rather than assumed.

**Not implemented in version one: uncertainty-based selection.** Uncertainty
estimates for protein models are not reliably calibrated, and uncertainty-driven
selection does not consistently beat simpler approaches.

---

## 10. Budget arithmetic is explicit

Candidate wells, replicates, and control wells are tracked and reported
separately. "24 candidates" and "24 wells" are different statements, and the
export makes which one you have unambiguous.

---

## 11. Identity across rounds

Every selected construct gets a `design_id` derived from the campaign id and
the sequence. It is stable: the same sequence always gets the same id, in this
run and in any future run. That is what makes `assaypilot score-round` able to
match results back to the predictions that proposed them, even if the plate map
renamed everything.

---

## 12. Scoring the round, then retraining

When results arrive, predictions are scored **before** any retraining, because
afterwards the question "what did we predict?" can no longer be asked cleanly.
The scorecard reports rank correlation within the delivered batch, the hit rate
if a threshold exists, the best value found, and how it compares with the
campaign's previous best — together with the note that it cannot say what a
different selection would have produced, since those variants were not
measured.

Then the round is appended as a new campaign version. Nothing earlier changes.
