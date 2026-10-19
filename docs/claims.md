# What AssayPilot claims, and what it does not

This page is the project's honesty contract. Each claim says what is asserted,
and — where it matters — the code that enforces it and the test that checks it.
If you find a place where the software claims more than this page allows, that
is a bug; please open an issue.

---

## 1. Every prediction is tied to one assay, under one condition

**Claimed.** A trained predictor in AssayPilot predicts *the recorded outcome of
the named assay under the named condition of its campaign.*

**Not claimed.** That the prediction transfers to a different assay, a different
condition, or an intrinsic biophysical property. A cellular activity readout is
not reinterpreted as a catalytic constant. A protein language model's sequence
likelihood is not treated as a measurement of function.

**Enforced by.** [`Campaign`](../src/assaypilot/campaign.py) requires an assay
name, units, objective, and condition label, and every manifest, export, and
report carries them. Measurements whose `condition_id` differs from the
campaign's are excluded with the reason "create a separate campaign for it
rather than pooling the measurements"
([`load_measurements`](../src/assaypilot/data/loaders.py)).

**Tested by.** `tests/test_loaders.py::test_other_conditions_are_excluded_not_pooled`.

---

## 2. Evidence comes from held-out measurements of this campaign

**Claimed.** The metrics shown come from evaluating on measurements the model
did not train on, under a protocol that resembles the next experimental step.

**Not claimed.** That performance on a public benchmark transfers to your assay.
Public benchmarks are used to choose credible starting points, not to assert
local performance.

**Enforced by.** [`choose_split_plan`](../src/assaypilot/evaluation/splits.py)
prefers round splits, then mutation-distance, then mutation-pattern, then a
random split — and labels a random split as an interpolation diagnostic in the
report, the manifest, and the recommendation's reasons.

**Tested by.** `tests/test_splits.py`, `tests/test_compare.py::test_random_split_only_is_labelled_exploratory`.

---

## 3. Replicates and identical sequences never straddle a split

**Claimed.** Grouping is by sequence hash. All measurements of the same sequence
are in the same partition, including replicates and a variant remeasured in a
later round.

**Enforced by.** `_check_no_group_overlap` runs inside every split builder and
raises if the invariant is broken; a sequence remeasured in a later round is
removed from the *evaluation* side and stays with the training rounds, which the
split description states.

**Tested by.** `tests/test_leakage.py::test_no_sequence_appears_in_both_partitions`,
`tests/test_splits.py::test_round_split_partitions_are_disjoint_by_sequence`.

---

## 4. Preprocessing and hyperparameters are fitted inside training folds

**Claimed.** Scaling, dimensionality reduction, the one-hot vocabulary, and the
ridge penalty are fitted on training data only. Outer held-out examples estimate
performance and are never used to tune the model reported on that same split.

**Enforced by.** Everything learnable lives inside the scikit-learn pipeline
fitted in `Method.fit`
([`assay_trained.py`](../src/assaypilot/models/assay_trained.py),
[`baseline.py`](../src/assaypilot/models/baseline.py)); hyperparameters are
chosen by a grouped inner cross-validation of the training rows.

**Tested by.** `tests/test_leakage.py::test_scaler_is_fitted_inside_the_training_fold`,
`::test_onehot_vocabulary_excludes_unseen_substitutions`,
`::test_methods_are_fitted_only_on_training_rows`.

---

## 5. A numerical lead is not a winner

**Claimed.** A more complex method is adopted only when the paired bootstrap 95%
interval for its improvement on the primary metric lies entirely above zero.
Otherwise the simpler supported method is used, and the report says the methods
were not distinguishable.

**Not claimed.** That the chosen method is the best possible method, or that the
comparison would hold on more data.

**Enforced by.** `_decide` in
[`compare.py`](../src/assaypilot/evaluation/compare.py). The rule is printed
verbatim in the report and stored in the manifest.

**Tested by.** `tests/test_compare.py::test_indistinguishable_methods_keep_the_simpler_one`,
`::test_a_clearly_better_method_is_adopted`.

---

## 5a. A metric that cannot separate methods is not used to choose one

**Claimed.** Hit rate drives the decision only when it can discriminate: a
threshold declared in advance, a held-out partition at least twice the batch
size, no more than 25% of measured variants already clearing the threshold,
and at least five hits held out. Otherwise rank correlation over the whole
held-out set is used, and the report, the warnings and the manifest all say
which condition failed.

**Why.** A success threshold that most measured variants already clear makes
every method look good — random selection included — because the base rate is
the floor and 1 is the ceiling. Public data makes the failure concrete: in
`CCDB_ECOLI_Tripathi_2016`, ties at a floor value put 79% of the pool above
the top decile, and random selection scored 131.8 hits out of 168 wells. A
number like that is not evidence of anything.

**Enforced by.** `choose_primary_metric` and `MeasurementSet.hit_base_rate`;
`plan_batch` raises a warning whenever a declared threshold could not be used.

**Tested by.** `tests/test_compare.py::test_a_threshold_most_variants_clear_cannot_drive_the_decision`,
`::test_too_few_held_out_hits_cannot_drive_the_decision`,
`tests/test_pipeline.py::test_a_threshold_most_variants_clear_is_flagged_in_the_plan`,
`::test_a_usable_threshold_still_drives_the_decision`.

---

## 5b. "Not distinguishable" says what could have been detected

**Claimed.** Whenever the comparison reports that methods could not be told
apart, it also reports the smallest difference the evaluation could have
resolved — half the width of the paired bootstrap interval — and the number of
distinct held-out sequences behind it.

**Why.** "No method was distinguishable" conflates two findings that call for
opposite actions: the methods are equivalent, or this dataset cannot tell. On
the bundled example the figure is 0.115, meaning a genuine eleven-point
improvement in hit rate would have been reported exactly the same way. Without
the figure a researcher cannot know which situation they are in.

**Not claimed.** That the figure is a formal power calculation against a
specified alternative. It is the resolution of this evaluation's own
resampling interval, which is what the decision rule reads.

**Enforced by.** `PairedComparison.resolution` and
`Recommendation.power_statement`, printed by the CLI, shown in the report, and
stored in the manifest.

**Tested by.** `tests/test_compare.py::test_a_null_result_says_what_it_could_have_detected`,
`::test_the_resolution_tightens_with_more_held_out_data`,
`::test_an_adopted_improvement_is_reported_as_exceeding_the_resolution`.

---

## 6. Weak evidence is labelled, not dressed up

**Claimed.** A plan is labelled `supported` only when the split was an
extrapolation protocol, at least 20 distinct held-out sequences were available,
and the chosen method beat chance **on the metric that drove the decision** —
zero for a rank correlation, and what random selection achieved on the same
held-out set for a hit rate. Otherwise it is
`exploratory`, in the report, the exports (`evidence_level` column), and the
manifest.

**Enforced by.** `MIN_TEST_SEQUENCES_FOR_SUPPORT`, `_chance_level`, and the
evidence checks in `_decide`. Every split is scored at the same budget `k`, so
the averaged metric is one quantity rather than a mixture, and the "is the top
batch a large share of the held-out set?" guard is applied to the smallest
split rather than to a pooled row count.

**Tested by.** `tests/test_compare.py::test_a_backwards_method_is_never_supported_under_a_hit_rate_metric`,
`::test_every_split_is_scored_at_the_same_budget`, `tests/test_pipeline.py`.

---

## 7. Uncertainty is rank variability, not calibration

**Claimed.** Bootstrap refits show how sensitive the candidate ranking is to the
measurements collected so far.

**Not claimed.** A calibrated probability of success, or a calibrated prediction
interval. Version one does not display intervals as calibrated, and does not
select candidates by uncertainty. Published benchmarking of uncertainty
quantification for protein engineering finds no uncertainty method or selection
policy that wins across datasets and distribution shifts (Greenman, Amini &
Yang, 2025).

**Enforced by.** `RANK_VARIABILITY_CAVEAT` travels with every variability table;
`interval_coverage` exists so a future release can *measure* calibration before
claiming it.

**Tested by.** `tests/test_selection.py::test_rank_variability_reports_stability_not_probability`.

---

## 8. The pretrained score is additive, and is a score

**Claimed.** ESM-2 masked marginals score a variant as the sum of single-site
log-ratios.

**Not claimed.** That the sum models epistasis, or that it is a predicted assay
value. The `predicted_value` column stays empty for this method; only `score` is
filled.

**Enforced by.** `MethodInfo.caveats` on
[`PretrainedScoreMethod`](../src/assaypilot/models/pretrained.py), reproduced in
the report; the pipeline only fills `predicted_value` for label-using methods.

---

## 9. Coverage is checked, not assumed

**Claimed.** For every candidate, AssayPilot reports its distance to the nearest
measured variant, how many of its substitutions have never been measured, and
whether it lies inside the region the evaluation covered. "The region the
evaluation covered" is measured per split, against that split's own training
set: pooling training rows across splits would put a sequence on both sides and
make the covered region look far narrower than it is.

**Not claimed.** That the reported metrics apply to candidates outside that
region. Those candidates are flagged, and `--restrict-to-evaluated` removes them
from selection.

**Tested by.** `tests/test_selection.py::test_coverage_flags_candidates_beyond_the_evaluated_region`,
`tests/test_pipeline.py::test_coverage_is_measured_against_each_split_s_own_training_set`,
`::test_unseen_substitutions_are_counted_against_all_measurements`.

---

## 10. Nothing is dropped or repaired silently

**Claimed.** Every row that does not reach the model is recorded with a code, a
reason, and its source row number, and is exported in `exclusions.csv`. Mutation
notation whose stated wild-type residue does not match the reference is an
error, never renumbered. Outliers are never removed on the tool's initiative.
Censored values are kept in the data and excluded from regression, not treated
as exact — including after a round trip through the campaign store, and
whichever entry point is used, since `plan_batch` removes them itself rather
than trusting the caller to. Each removed row is recorded as a
`censored_value` exclusion with its source row, not merely summarised in a
note.

**Tested by.** `tests/test_loaders.py` (whole module, including
`::test_for_regression_records_every_dropped_row`),
`tests/test_sequences.py::test_parse_mutation_string_rejects_wrong_wildtype`,
`tests/test_pipeline.py::test_censored_measurements_never_reach_the_model`,
`tests/test_rounds.py::test_store_preserves_censoring_across_a_round_trip`.

---

## 10a. A transform moves the threshold and the units with it

**Claimed.** When values are transformed, the campaign's success threshold is
mapped through the same transform before any hit is counted, and exported
predictions are labelled in the transformed space (`log10(RFU)`, not `RFU`).

**Not claimed.** That a transformed prediction can be read as a raw assay
value. AssayPilot does not back-transform predictions, because the honest
retransformation depends on assumptions it has not checked.

**Enforced by.** `MeasurementSet.threshold_in_value_space`,
`MeasurementSet.value_units`, and `MeasurementSet.is_hit_array`; the evaluation
and the round scorecard both go through them. A threshold the transform cannot
express — a non-positive threshold under a log — disables hit rates and says
so, rather than producing zero hits.

**Tested by.** `tests/test_loaders.py::test_transform_maps_the_success_threshold_into_value_space`,
`::test_a_threshold_a_transform_cannot_express_yields_no_hits`,
`tests/test_pipeline.py::test_predicted_values_are_labelled_with_the_transformed_units`.

---

## 11. A run can be reproduced from its manifest

**Claimed.** The manifest records input checksums, transformations, partition
assignments, seeds, chosen hyperparameters, the pinned checkpoint revision,
package versions, every prediction, every exclusion, and the selected batch.
Reruns with the same inputs and configuration produce the same fingerprint.

**Not claimed.** That a checkpoint *name* is enough. An unpinned checkpoint is
recorded as `"unpinned"` and logged as a warning.

**Tested by.** `tests/test_pipeline.py::test_plan_is_deterministic`,
`::test_manifest_records_what_reproduction_needs`.

---

## 12. History is append-only

**Claimed.** Adding a round creates a new campaign version. Runs and selections
are inserted once and never updated, so an old plan still reads as it did when
it was made — including predictions that turned out to be wrong.

**Tested by.** `tests/test_rounds.py::test_adding_a_round_creates_a_new_version_without_touching_history`.

---

## 13. What is *not* established

- **No performance claim.** This repository ships no benchmark numbers. Whether
  AssayPilot improves a hit rate on any assay is an outcome to measure. The
  tooling to measure it is `assaypilot benchmark`; the protocol is in
  [benchmarks.md](benchmarks.md).
- **No novelty claim.** EVOLVEpro, OpenProtein, and ProteusAI already cover much
  of this category. AssayPilot does not assert they lack evaluation.
- **No "unseen by the model" claim.** Published assay sequences may overlap the
  pretraining corpora of protein language models. Results on public datasets are
  never described as evaluation on sequences a model has never seen.
- **No claim for the physicochemical representation.** It exists so the workflow
  runs without model weights. It is not a protein language model and is labelled
  that way everywhere it appears.
- **No clinical, diagnostic, or safety claim.** AssayPilot plans laboratory
  experiments on a protein you supply, in an assay you define. It makes no
  statement about any molecule's behaviour outside that assay.
