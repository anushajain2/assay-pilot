"""ESM-2 embeddings and masked marginals, against a real checkpoint.

Every test here is marked ``slow`` and skipped unless torch, transformers, and
a downloadable checkpoint are available. The rest of the suite deliberately
runs without them, so these are opt-in::

    pytest -m slow
    ASSAYPILOT_TEST_CHECKPOINT=facebook/esm2_t12_35M_UR50D pytest -m slow

The default checkpoint is the 8M one: it is the smallest ESM-2 release, which
keeps the download and the runtime small. These tests check the *mechanics* —
tokenizer offsets, pooling, additivity, caching, determinism — not predictive
quality, which is what `assaypilot benchmark` is for.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from assaypilot.features.esm import esm_available

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not esm_available(), reason="torch and transformers are not installed"),
]

CHECKPOINT = os.environ.get("ASSAYPILOT_TEST_CHECKPOINT", "facebook/esm2_t6_8M_UR50D")
REFERENCE = "MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNL"


def _mutate(*substitutions: tuple[int, str]) -> str:
    residues = list(REFERENCE)
    for position, residue in substitutions:
        residues[position - 1] = residue
    return "".join(residues)


@pytest.fixture(scope="module")
def backend():
    from assaypilot.features.esm import ESM2Backend

    return ESM2Backend(CHECKPOINT)


@pytest.fixture(scope="module")
def scorer():
    from assaypilot.features.esm import MaskedMarginalScorer

    return MaskedMarginalScorer(REFERENCE, CHECKPOINT)


# ----------------------------------------------------------------- tokenizer


def test_residue_positions_line_up_with_token_indices(backend):
    """The masked-marginal scorer masks token ``i`` for 1-based residue ``i``.

    If the tokenizer ever stopped prefixing exactly one <cls> token, every
    mutation would be scored at the wrong position — silently, and with
    plausible-looking numbers. This is the assumption that makes the scorer
    correct, so it is checked directly.
    """
    tokenizer = backend.holder.tokenizer
    encoded = tokenizer([REFERENCE], return_tensors="pt")
    ids = encoded["input_ids"][0].tolist()

    assert len(ids) == len(REFERENCE) + 2  # <cls> ... <eos>
    tokens = tokenizer.convert_ids_to_tokens(ids)
    assert tokens[0] == tokenizer.cls_token
    assert tokens[-1] == tokenizer.eos_token
    for index, residue in enumerate(REFERENCE):
        assert tokens[index + 1] == residue


def test_every_amino_acid_is_in_the_vocabulary(backend):
    from assaypilot.sequences import AMINO_ACIDS

    tokenizer = backend.holder.tokenizer
    unknown = tokenizer.unk_token_id
    missing = [aa for aa in AMINO_ACIDS if tokenizer.convert_tokens_to_ids(aa) == unknown]
    assert missing == []


# ----------------------------------------------------------------- embedding


def test_embeddings_have_the_expected_shape_and_are_finite(backend):
    variants = [REFERENCE, _mutate((3, "G")), _mutate((5, "W"))]
    embeddings = backend.embed(variants)

    assert embeddings.shape[0] == len(variants)
    assert embeddings.shape[1] > 0
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()


def test_different_variants_get_different_embeddings(backend):
    embeddings = backend.embed([REFERENCE, _mutate((3, "G")), _mutate((20, "K"))])
    assert not np.allclose(embeddings[0], embeddings[1])
    assert not np.allclose(embeddings[1], embeddings[2])


def test_embedding_is_deterministic(backend):
    variants = [REFERENCE, _mutate((7, "D"))]
    np.testing.assert_allclose(backend.embed(variants), backend.embed(variants))


def test_pooling_ignores_padding(backend):
    """A sequence must embed identically alone and batched with a longer one.

    Mean-pooling over padded positions would make a variant's representation
    depend on what else happened to be in its batch, which would quietly make
    results depend on batch size.
    """
    short = REFERENCE[:12]
    alone = backend.embed([short])[0]
    batched = backend.embed([short, REFERENCE])[0]
    np.testing.assert_allclose(alone, batched, atol=1e-5)


# ----------------------------------------------------------- masked marginals


def test_the_reference_scores_exactly_zero(scorer):
    """The wild type has no substitutions, so its summed score must be 0."""
    assert scorer.score([REFERENCE])[0] == 0.0


def test_multi_substitution_scores_are_the_sum_of_single_site_scores(scorer):
    """The additive approximation the method documents, verified rather than assumed."""
    single_a = _mutate((3, "G"))
    single_b = _mutate((20, "K"))
    double = _mutate((3, "G"), (20, "K"))

    scores = scorer.score([single_a, single_b, double])
    assert scores[2] == pytest.approx(scores[0] + scores[1], abs=1e-4)


def test_scores_differ_between_substitutions(scorer):
    scores = scorer.score([_mutate((3, "G")), _mutate((3, "P")), _mutate((20, "K"))])
    assert len(set(np.round(scores, 6))) == 3


def test_scoring_is_deterministic(scorer):
    variants = [_mutate((5, "W")), _mutate((9, "D"))]
    np.testing.assert_allclose(scorer.score(variants), scorer.score(variants))


# -------------------------------------------------- reproducibility plumbing


def test_the_checkpoint_revision_is_recorded(backend):
    """A checkpoint name alone cannot reproduce a result; the commit can."""
    backend.prepare()
    fingerprint = backend.fingerprint()

    assert fingerprint["checkpoint"] == CHECKPOINT
    assert fingerprint["is_language_model"] is True
    assert fingerprint["revision_source"] in {"requested", "resolved", "unpinned"}
    if fingerprint["revision_source"] != "unpinned":
        assert len(fingerprint["revision"]) == 40


def test_the_cache_namespace_does_not_depend_on_the_model_being_loaded():
    """Cold and warm runs must agree, or every cached vector is recomputed."""
    from assaypilot.features.esm import ESM2Backend

    unloaded = ESM2Backend(CHECKPOINT)
    unloaded.prepare()

    loaded = ESM2Backend(CHECKPOINT)
    loaded.embed([REFERENCE])

    assert unloaded.fingerprint_key() == loaded.fingerprint_key()


def test_a_pinned_revision_is_reported_as_requested(backend):
    from assaypilot.features.esm import ESM2Backend

    backend.prepare()
    revision = backend.effective_revision
    if revision is None:
        pytest.skip("the checkpoint revision could not be resolved in this environment")

    pinned = ESM2Backend(CHECKPOINT, revision=revision)
    fingerprint = pinned.fingerprint()
    assert fingerprint["revision"] == revision
    assert fingerprint["revision_source"] == "requested"
    # Pinning the revision the backend already resolved must not change the
    # cache namespace, or pinning would throw away every cached embedding.
    assert pinned.fingerprint_key() == backend.fingerprint_key()


def test_embeddings_are_cached_by_sequence(tmp_path):
    from assaypilot.features.base import EmbeddingCache, embed_sequences
    from assaypilot.features.esm import ESM2Backend

    cache = EmbeddingCache(tmp_path / "cache")
    backend = ESM2Backend(CHECKPOINT)
    variants = [REFERENCE, _mutate((3, "G"))]

    first = embed_sequences(backend, variants, cache=cache)
    assert cache.misses == 2 and cache.hits == 0

    second = embed_sequences(backend, variants, cache=cache)
    assert cache.hits == 2
    np.testing.assert_allclose(first, second)


# --------------------------------------------------------- the whole workflow


@pytest.mark.parametrize("policy", ["top_ranked"])
def test_a_full_plan_runs_with_esm2(tmp_path, policy):
    """The three-method suite, end to end, with a real protein language model."""
    from assaypilot import load_candidates, load_measurements, plan_batch
    from assaypilot.datasets import simulate_campaign
    from assaypilot.models.registry import SuiteConfig
    from assaypilot.reporting.exports import export_plan

    simulated = simulate_campaign(
        n_measured=40, n_candidates=20, n_rounds=2, n_variable_positions=8, seed=2
    )
    campaign = simulated.campaign
    measurements = load_measurements(simulated.measurements, campaign, strict=False)
    candidates = load_candidates(simulated.candidates, campaign, strict=False)

    plan = plan_batch(
        campaign,
        candidates,
        measurements,
        suite=SuiteConfig(
            representation="esm2",
            esm_checkpoint=CHECKPOINT,
            cache_dir=str(tmp_path / "cache"),
        ),
        policy=policy,
        n_bootstrap=20,
        seed=0,
    )

    # All three methods ran: nothing was reported as unavailable.
    assert plan.evaluation is not None
    assert plan.evaluation.unavailable == {}
    assert set(plan.evaluation.summaries) == {
        "baseline_onehot_ridge",
        "pretrained_esm_masked_marginal",
        "assay_trained_embedding_ridge",
    }

    # The checkpoint and its revision are recorded whichever method wins.
    representation = plan.manifest.config["representation"]
    assert representation["checkpoint"] == CHECKPOINT

    written = export_plan(plan, tmp_path / "out")
    assert written["report"].exists()


def test_the_pretrained_score_exports_no_predicted_value(tmp_path):
    """A model score on an arbitrary scale must not be exported as an assay value."""
    import pandas as pd

    from assaypilot import load_candidates, load_measurements, plan_batch
    from assaypilot.datasets import simulate_campaign
    from assaypilot.models.registry import SuiteConfig

    simulated = simulate_campaign(
        n_measured=40, n_candidates=20, n_rounds=2, n_variable_positions=8, seed=4
    )
    campaign = simulated.campaign
    candidates = load_candidates(simulated.candidates, campaign, strict=False)
    measurements = load_measurements(simulated.measurements, campaign, strict=False)

    plan = plan_batch(
        campaign,
        candidates,
        measurements,
        suite=SuiteConfig(
            representation="esm2",
            esm_checkpoint=CHECKPOINT,
            include_baseline=False,
            include_assay_trained=False,
            cache_dir=str(tmp_path / "cache"),
        ),
        n_bootstrap=10,
        seed=0,
    )

    assert plan.method_key == "pretrained_esm_masked_marginal"
    assert plan.ranking["score"].notna().all()
    assert pd.isna(plan.ranking["predicted_value"]).all()
    assert set(plan.ranking["predicted_value_units"]) == {""}


def test_a_pretrained_only_plan_is_labelled_unvalidated(tmp_path):
    """With no measurements, the label-free score can still rank — unvalidated."""
    from assaypilot import load_candidates, plan_batch
    from assaypilot.datasets import simulate_campaign
    from assaypilot.models.registry import SuiteConfig

    simulated = simulate_campaign(
        n_measured=20, n_candidates=15, n_rounds=1, n_variable_positions=6, seed=6
    )
    campaign = simulated.campaign
    candidates = load_candidates(simulated.candidates, campaign, strict=False)

    plan = plan_batch(
        campaign,
        candidates,
        None,
        suite=SuiteConfig(
            representation="esm2",
            esm_checkpoint=CHECKPOINT,
            cache_dir=str(tmp_path / "cache"),
        ),
    )

    assert plan.method_key == "pretrained_esm_masked_marginal"
    assert plan.evidence_level == "none"
    assert plan.evaluation is None
    assert plan.ranking["predicted_value"].isna().all()
    assert any("no claim of assay-specific validation" in n.lower() for n in plan.notes)
    assert "exploratory" in plan.headline().lower()
