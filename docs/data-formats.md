# Data formats

Everything AssayPilot reads and writes, column by column.

---

## Measurements

CSV, TSV, or Parquet. Column names are matched case-insensitively, and common
aliases are accepted (`id`, `variant`, `seq`, `fitness`, `activity`, `score`,
`round`, `rep`, `plate`).

| Column | Required | Type | Meaning |
| --- | --- | --- | --- |
| `variant_id` | yes | text | Identifier, unique per distinct sequence. The same id with two different sequences is a blocking error. |
| `sequence` | one of these | text | Full-length sequence, same length as the reference, standard amino acids only. |
| `mutations` | one of these | text | `A12G:L30P`, `A12G, L30P`, `A12G+L30P`, or `WT`. The stated wild-type residue must match the reference at that position. |
| `value` | yes | number or censored | The measured outcome. `<0.01` / `>5` are read as censored. |
| `condition_id` | recommended | text | Experimental condition. Rows whose condition differs from the campaign's are excluded, not pooled. |
| `round_id` | recommended | text | Round label. Two or more rounds unlock round-based evaluation. |
| `replicate_id` | optional | text | Replicate label. Replicates always share an evaluation partition. |
| `plate_id` | optional | text | Plate label, carried through for review. |

Example:

```csv
variant_id,sequence,value,condition_id,round_id,replicate_id,plate_id
var0001,MKAILVVLL...,1.84,37C_pH7.4,R1,rep1,P1
var0001,MKAILVVLL...,1.79,37C_pH7.4,R1,rep2,P1
var0002,MKAILVVLL...,0.42,37C_pH7.4,R1,rep1,P1
```

With mutation notation instead of sequences:

```csv
variant_id,mutations,value,condition_id,round_id
var0001,A12G,1.84,37C_pH7.4,R1
var0002,A12G:L30P,0.42,37C_pH7.4,R1
var0003,WT,1.00,37C_pH7.4,R1
```

### Censored values

`<` and `>` prefixes mark a value as censored at a limit. The number is kept,
the censoring direction is recorded, and the row is **excluded from model
fitting** in version one rather than being treated as exact. It still appears
in the data review and in the exported tables.

### Transformations

`--transform log10|log2|ln|zscore` applies a transformation to `value` while
keeping the original in `value_raw`. The transformation is recorded in the
manifest and named in the report. A log transform on zero or negative values is
refused rather than silently offset.

---

## Candidates

CSV, TSV, Parquet, or FASTA.

| Column | Required | Meaning |
| --- | --- | --- |
| `candidate_id` | recommended | Unique identifier. Generated as `cand00001…` if absent. |
| `sequence` | one of these | Full-length sequence, reference length. |
| `mutations` | one of these | Mutation notation, checked against the reference. |

```csv
candidate_id,sequence
cand0001,MKAILVVLL...
cand0002,MKAILVVLL...
```

```fasta
>cand0001 round-4 design
MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNL
>cand0002
MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNL
```

Duplicate sequences inside the pool are excluded (first occurrence kept), so a
batch never spends two wells on the same construct.

---

## `campaign.json`

```json
{
  "campaign_id": "bglb-thermostability",
  "reference_sequence": "MKAILVVLL...",
  "assay_name": "residual activity after 50C, 30 min",
  "assay_units": "fraction of unheated control",
  "objective": "maximize",
  "condition_id": "50C_30min_lysate",
  "description": "",
  "success_threshold": 0.6,
  "created_at": "2026-10-18T09:12:00+00:00",
  "selection": {
    "batch_size": 24,
    "locked_positions": [145],
    "max_substitutions": 4,
    "allow_measured_repeats": false,
    "diversity_min_distance": 2,
    "control_sequences": ["WT", "no-enzyme"],
    "replicates_per_candidate": 1
  }
}
```

---

## Outputs

### `selected_variants.csv`

| Column | Meaning |
| --- | --- |
| `design_id` | Immutable identifier derived from campaign id + sequence. Use it to label wells and to match results back. |
| `candidate_id` | The id from your candidate file. |
| `selection_rank` | 1 = first chosen. |
| `sequence`, `mutations`, `n_substitutions` | The construct. |
| `score` | The method's ranking value (higher = better under the campaign objective). Scale may be arbitrary. |
| `predicted_value` | The score expressed in assay units. **Empty** for methods that produce only a score. |
| `predicted_value_units` | Units of `predicted_value`, empty when there is none. |
| `observed_value` | Measured value, filled only if this sequence has already been measured. |
| `min_distance_to_measured` | Substitutions from the nearest measured variant. |
| `n_unseen_substitutions` | Substitutions never seen in any measurement. |
| `within_evaluated_range` | Whether the evaluation covered variants this far out. |
| `rank_std`, `top_k_frequency` | Ranking stability across bootstrap refits, when requested. Not a probability of success. |
| `selection_reason` | Why this candidate is in the batch. |
| `method`, `evidence_level` | Which method chose it, and how strong the evidence was. |
| `campaign_id`, `run_id` | Provenance. |

### `all_predictions.csv`

Every candidate, eligible or not, with `eligible`, `rank_among_eligible`, and
the same score/coverage columns.

### `exclusions.csv`

| Column | Meaning |
| --- | --- |
| `row` | Source row number in the input file (header is row 1). |
| `identifier` | The variant or candidate id. |
| `code` | Machine-readable reason (see below). |
| `severity` | `blocking` (fix the file) or `excluded` (not usable here). |
| `detail` | The reason in words. |

Codes: `missing_sequence`, `invalid_characters`, `length_mismatch`,
`mutation_parse_error`, `missing_value`, `non_numeric_value`, `other_condition`,
`missing_id`, `duplicate_id`, `duplicate_sequence`, `already_measured`,
`locked_position_changed`, `exceeds_max_substitutions`, `censored_value`,
`identical_to_reference`, `restricted`.

### `candidate_coverage.csv`

Per-candidate distance and novelty, as described above.

### `method_comparison.csv`

Long format: one row per method per split (`scope = split`) plus one aggregate
row per method (`scope = aggregate`), with `spearman`, `hit_rate_at_k`,
`best_at_k`, `mean_at_k`, `pearson`, `n_test`, `k`.

### `run_manifest.json`

| Section | Contents |
| --- | --- |
| `campaign` | The full campaign definition used. |
| `inputs` | Path, name, SHA-256, and size of each input file. |
| `config` | Selection settings, suite configuration, policy, split policy, seed, bootstrap count, chosen hyperparameters, the method used. |
| `runtime` | Observations that do not affect results, such as embedding-cache hits. |
| `data` | Row counts, rounds, transformation, and the full validation report for each input. |
| `evaluation` | Split plan with **explicit train/test row indices**, per-split and aggregate metrics, paired comparisons, the recommendation and its reasons, evidence level, rank variability. |
| `coverage` | Pool-level coverage summary and warnings. |
| `selection` | Policy, shortfall, filters applied, budget. |
| `predictions` | Every candidate with its score and eligibility. |
| `exclusions` | Every excluded row. |
| `selected` | The batch as exported. |
| `environment` | Python version, platform, and versions of numpy, pandas, scikit-learn, scipy, torch, transformers. |
| `warnings`, `notes` | Everything the report showed. |

`RunManifest.fingerprint()` hashes campaign + inputs + config + environment.
Two runs with the same fingerprint used the same inputs and settings;
timestamps, run ids, and cache statistics are excluded so a deterministic rerun
matches.

### `report.html`

Self-contained, no external assets: the batch, the budget, the evidence, the
paired comparisons, the selection rule and its reasons, coverage, exclusions,
warnings, and the reproducibility table.

---

## Round results

To score a plan, supply the next round's measurements in the normal
measurements format. Matching is by `design_id`, which AssayPilot recomputes
from each measured **sequence** — so the match holds however the wells were
labelled, and naming a variant after its `design_id` is a convenience, not a
requirement:

```csv
variant_id,sequence,value,condition_id,round_id
de9fbcf75f2b,MKAILVVLL...,2.31,37C_pH7.4,R4
d1bf9a6cdc15,MKAILVVLL...,1.88,37C_pH7.4,R4
```

Because the identifier is derived from the sequence and the campaign id, a
construct that reappears in a later round is recognised automatically.
