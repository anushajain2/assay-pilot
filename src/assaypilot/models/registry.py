"""Building the fixed comparison suite.

The suite is deliberately small and fixed: a simple baseline, a label-free
pretrained score, and an assay-trained model on frozen representations. Every
method receives the same eligible training data and is evaluated on the same
examples.

Keeping the suite small is a methodological choice, not a limitation of the
implementation. Searching across many models until one appears to win inflates
the apparent performance of whichever model happens to fit the noise of a
small assay dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..campaign import Campaign
from ..features.base import EmbeddingBackend, EmbeddingCache
from ..features.esm import DEFAULT_CHECKPOINT, ESM2Backend
from ..features.physchem import PhysicochemicalBackend
from .assay_trained import EmbeddingRidgeMethod
from .base import Method
from .baseline import OneHotRidgeBaseline
from .pretrained import PretrainedScoreMethod

Representation = Literal["esm2", "physchem"]


@dataclass
class SuiteConfig:
    """How the comparison suite is built for a run.

    Attributes
    ----------
    representation:
        ``"esm2"`` (default) uses a protein language model and needs the
        optional ``esm`` extra. ``"physchem"`` uses the dependency-free
        physicochemical representation, which is clearly labelled as not a
        protein language model wherever it appears.
    esm_checkpoint, esm_revision:
        The checkpoint and its pinned revision. A checkpoint name on its own is
        not enough to reproduce a result, so the revision is recorded in every
        manifest and a warning is logged when it is left unpinned.
    """

    representation: Representation = "esm2"
    esm_checkpoint: str = DEFAULT_CHECKPOINT
    esm_revision: str | None = None
    device: str | None = None
    physchem_bins: int = 16
    cache_dir: str | None = None
    use_cache: bool = True
    include_baseline: bool = True
    include_pretrained: bool = True
    include_assay_trained: bool = True
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    pca_components: tuple[int | None, ...] = (None, 16, 64)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["alphas"] = list(self.alphas)
        data["pca_components"] = list(self.pca_components)
        return data


def build_backend(config: SuiteConfig) -> EmbeddingBackend:
    """Create the representation backend named by ``config``."""
    if config.representation == "physchem":
        return PhysicochemicalBackend(n_bins=config.physchem_bins)
    return ESM2Backend(config.esm_checkpoint, revision=config.esm_revision, device=config.device)


def build_cache(config: SuiteConfig) -> EmbeddingCache:
    """Create the embedding cache described by ``config``."""
    root = Path(config.cache_dir) if config.cache_dir else None
    return EmbeddingCache(root, enabled=config.use_cache)


def build_methods(
    campaign: Campaign,
    config: SuiteConfig | None = None,
    *,
    cache: EmbeddingCache | None = None,
) -> list[Method]:
    """Build the comparison suite for a campaign, in increasing complexity order."""
    config = config or SuiteConfig()
    cache = cache if cache is not None else build_cache(config)
    methods: list[Method] = []

    if config.include_baseline:
        methods.append(OneHotRidgeBaseline(campaign.reference_sequence, alphas=config.alphas))

    if config.include_pretrained:
        methods.append(
            PretrainedScoreMethod(
                campaign.reference_sequence,
                checkpoint=config.esm_checkpoint,
                revision=config.esm_revision,
                device=config.device,
            )
        )

    if config.include_assay_trained:
        methods.append(
            EmbeddingRidgeMethod(
                build_backend(config),
                cache=cache,
                alphas=config.alphas,
                n_components=config.pca_components,
            )
        )

    return methods


def split_by_availability(methods: list[Method]) -> tuple[list[Method], dict[str, str]]:
    """Separate usable methods from those whose backend is missing.

    An unavailable method is never dropped quietly: its reason is returned and
    shown in the report, so a comparison that ran without ESM-2 says so.
    """
    usable: list[Method] = []
    unavailable: dict[str, str] = {}
    for method in methods:
        ok, reason = method.is_available()
        if ok:
            usable.append(method)
        else:
            unavailable[method.info.key] = reason
    return usable, unavailable
