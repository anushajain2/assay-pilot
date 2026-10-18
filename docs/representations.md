# Representations

AssayPilot turns sequences into numbers in three ways. Which one is used, and
with which exact checkpoint, is recorded in every run manifest — because a
model *name* is not enough to reproduce a result.

---

## 1. One-hot substitutions (the baseline's features)

Binary indicators for each `(position, mutant residue)` pair **seen during
fitting**, plus the total number of substitutions.

- Dimension: the number of distinct substitutions in the training fold (+1).
- A substitution absent from the training fold is represented as all-zeros, so
  the baseline predicts near the training mean for such variants. The share of
  such substitutions is reported (`unseen_substitution_fraction`).
- The vocabulary is fitted inside the pipeline, so cross-validation cannot leak
  held-out substitutions into training.

Use it as the honesty check: if a protein-model representation cannot beat
this, it is not earning its compute on this assay.

---

## 2. ESM-2 embeddings (`--representation esm2`, the default)

Frozen mean-pooled hidden states from an ESM-2 checkpoint. Special tokens
(`<cls>`, `<eos>`, padding) are excluded from pooling. The protein model is
never fine-tuned; only the ridge head on top is fitted, inside training folds.

| Checkpoint | Notes |
| --- | --- |
| `facebook/esm2_t6_8M_UR50D` | Fastest, weakest representation |
| `facebook/esm2_t12_35M_UR50D` | **Default** |
| `facebook/esm2_t30_150M_UR50D` | Middle tier |
| `facebook/esm2_t33_650M_UR50D` | GPU recommended |

Model size is a **compute tier, not an assumed improvement**: a bigger
checkpoint does not establish better performance on a particular assay. If you
want to know whether 650M helps on yours, run the comparison.

**Pin the revision.** `--esm-revision <commit>` records the exact checkpoint
revision in the manifest. Without it the manifest says `"unpinned"` and a
warning is logged, because a checkpoint can be updated under the same name.

```bash
assaypilot plan ... --representation esm2 \
  --esm-checkpoint facebook/esm2_t12_35M_UR50D \
  --esm-revision 08e4846e537d2f3bf0b2ac8e0a4d7e7c0e5e0c7f
```

### Masked-marginal scores

The pretrained method masks each mutated position of the reference and scores a
variant as the sum over its substitutions of
`log p(mutant) − log p(wild type)`.

Two properties travel with every such score:

- **Additive.** Summing single-site terms assumes substitutions act
  independently, so the score cannot represent epistasis. Variants with more
  than one substitution are flagged.
- **A score, not a value.** The scale is arbitrary; `predicted_value` stays
  empty for this method.

Computation cost is one forward pass per *mutated position* of the reference
(not per variant), cached for the run.

---

## 3. Physicochemical features (`--representation physchem`)

A dependency-free representation so that tests, CI, the bundled demo, and a
first look on a laptop work without downloading weights.

Seven residue property scales — hydropathy, volume, charge, polarity,
isoelectric point, helix propensity, aromaticity — standardised across the 20
amino acids, then pooled globally and in `n_bins` contiguous positional bins
(default 16). Dimension: `7 × (n_bins + 1)`, independent of sequence length;
the positional binning is what makes a substitution at residue 40 distinguish­
able from the same substitution at residue 300.

**It is not a protein language model.** Every report, method description, and
manifest entry that involves it says so. Do not present results obtained with
it as ESM-2 results.

Property values and their sources are in
[`src/assaypilot/features/physchem.py`](../src/assaypilot/features/physchem.py):
Kyte–Doolittle hydropathy, residue volumes, formal charge at pH 7, Grantham
polarity, isoelectric points, Pace–Scholtz helix propensities, and an
aromaticity indicator.

---

## The embedding cache

Embeddings depend only on the sequence and the representation configuration, so
caching them changes nothing about an evaluation.

- Location: `$ASSAYPILOT_CACHE_DIR`, else `$XDG_CACHE_HOME/assaypilot/embeddings`,
  else `~/.cache/assaypilot/embeddings`. Override with `--cache-dir`, disable
  with `--no-cache`.
- Key: `sha256(sequence)` under a namespace derived from the backend
  fingerprint (backend, checkpoint, revision, layer, pooling). Changing the
  checkpoint cannot reuse the wrong vectors.
- Entries are written to a temporary file and renamed, so an interrupted run
  leaves no half-written entry. An unreadable entry is treated as a miss, never
  as a failure.

---

## Adding your own representation

Implement `EmbeddingBackend`:

```python
from assaypilot.features.base import EmbeddingBackend

class MyBackend(EmbeddingBackend):
    name = "my-backend"

    def embed(self, sequences):
        ...  # -> np.ndarray of shape (len(sequences), n_features)

    def fingerprint(self):
        return {"backend": self.name, "version": 1, "is_language_model": False}
```

Then pass it to `EmbeddingRidgeMethod(MyBackend())`. The fingerprint must change
whenever the output would change — it is both the cache key and the
reproducibility record.
