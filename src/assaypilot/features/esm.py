"""ESM-2 embeddings and masked-marginal mutation scores.

Optional install::

    pip install "assaypilot[esm]"

Two things live here:

:class:`ESM2Backend`
    Frozen mean-pooled embeddings from an ESM-2 checkpoint, used as the
    representation for the assay-trained model.
:class:`MaskedMarginalScorer`
    The pretrained, label-free mutation score. For each mutated position the
    reference is masked and the score is ``log p(mutant) - log p(wild type)``.
    Scores for multiple substitutions are **summed**, which assumes additivity
    and therefore does not model epistasis. Every result carries that label.

The default checkpoint is ``facebook/esm2_t12_35M_UR50D``, the 35M-parameter
model. Larger checkpoints are a compute tier, not an assumed improvement:
model size alone does not establish better performance on a particular assay.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import BackendUnavailableError
from ..sequences import AMINO_ACIDS, substitutions
from .base import EmbeddingBackend

logger = logging.getLogger(__name__)

#: Checkpoints with a known parameter count, offered as compute tiers.
KNOWN_CHECKPOINTS: dict[str, str] = {
    "facebook/esm2_t6_8M_UR50D": "8M parameters, fastest, weakest representation",
    "facebook/esm2_t12_35M_UR50D": "35M parameters, default",
    "facebook/esm2_t30_150M_UR50D": "150M parameters",
    "facebook/esm2_t33_650M_UR50D": "650M parameters, GPU recommended",
}

DEFAULT_CHECKPOINT = "facebook/esm2_t12_35M_UR50D"


def esm_available() -> bool:
    """Whether torch and transformers can be imported in this environment."""
    try:  # pragma: no cover - depends on the environment
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _require_esm() -> tuple[Any, Any]:
    try:  # pragma: no cover - depends on the environment
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover
        raise BackendUnavailableError(
            "ESM-2 support needs torch and transformers. Install them with"
            " 'pip install \"assaypilot[esm]\"', or run with --representation physchem"
            " to use the dependency-free representation (which is not a protein"
            " language model)."
        ) from exc
    return torch, transformers


def _resolve_device(device: str | None) -> str:
    torch, _ = _require_esm()
    if device and device != "auto":
        return device
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"  # pragma: no cover - hardware dependent
    return "cpu"


class _ESMModelHolder:
    """Loads a checkpoint once and shares it between the backend and the scorer."""

    _cache: dict[tuple[str, str | None, str], Any] = {}

    def __init__(self, checkpoint: str, revision: str | None, device: str | None) -> None:
        torch, transformers = _require_esm()
        self.torch = torch
        self.checkpoint = checkpoint
        self.requested_revision = revision
        self.device = _resolve_device(device)

        key = (checkpoint, revision, self.device)
        cached = self._cache.get(key)
        if cached is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, revision=revision)
            model = transformers.AutoModelForMaskedLM.from_pretrained(checkpoint, revision=revision)
            model.eval()
            model.to(self.device)
            cached = (tokenizer, model)
            self._cache[key] = cached
        self.tokenizer, self.model = cached

        resolved = (
            revision
            or getattr(self.model.config, "_commit_hash", None)
            or getattr(self.model.config, "commit_hash", None)
        )
        self.resolved_revision = resolved
        if resolved is None:
            logger.warning(
                "The ESM checkpoint %s was loaded without a pinned revision. The run manifest"
                " will record 'unpinned', which is not sufficient to reproduce the result"
                " exactly. Pass --esm-revision <commit> to pin it.",
                checkpoint,
            )

    def encode(self, sequences: Sequence[str]):
        return self.tokenizer(list(sequences), return_tensors="pt", padding=True)


class ESM2Backend(EmbeddingBackend):
    """Frozen ESM-2 embeddings, mean-pooled over residue positions.

    Special tokens (``<cls>``, ``<eos>``, padding) are excluded from pooling.
    The model is never fine-tuned: only the regression head on top of these
    embeddings is fitted, and only inside training folds.
    """

    name = "esm2"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        revision: str | None = None,
        layer: int = -1,
        pooling: str = "mean",
        device: str | None = None,
    ) -> None:
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.checkpoint = checkpoint
        self.revision = revision
        self.layer = layer
        self.pooling = pooling
        self.device = device
        self._holder: _ESMModelHolder | None = None

    @property
    def holder(self) -> _ESMModelHolder:
        if self._holder is None:
            self._holder = _ESMModelHolder(self.checkpoint, self.revision, self.device)
        return self._holder

    @property
    def is_language_model(self) -> bool:
        return True

    @property
    def description(self) -> str:
        note = KNOWN_CHECKPOINTS.get(self.checkpoint, "")
        suffix = f" ({note})" if note else ""
        return f"frozen ESM-2 embeddings from {self.checkpoint}{suffix}, {self.pooling}-pooled"

    def embed(self, sequences: Sequence[str]) -> np.ndarray:
        holder = self.holder
        torch = holder.torch
        encoded = holder.encode(sequences)
        encoded = {k: v.to(holder.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = holder.model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[self.layer]

        mask = encoded["attention_mask"].clone()
        # Drop <cls> and the final <eos>/padding boundary from pooling.
        mask[:, 0] = 0
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        for row, length in enumerate(lengths.tolist()):
            mask[row, int(length)] = 0

        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return pooled.detach().cpu().numpy().astype(np.float32)

    def fingerprint(self) -> dict[str, Any]:
        resolved = self._holder.resolved_revision if self._holder is not None else self.revision
        return {
            "backend": self.name,
            "checkpoint": self.checkpoint,
            "revision": resolved or "unpinned",
            "layer": self.layer,
            "pooling": self.pooling,
            "is_language_model": True,
        }


class MaskedMarginalScorer:
    """Label-free ESM-2 mutation scores, computed by masked marginals.

    The reference sequence is masked one position at a time, giving a
    ``position x amino acid`` log-probability table. A variant's score is the
    sum over its substitutions of ``log p(mutant) - log p(wild type)`` at the
    masked position.

    Summing single-site terms is an approximation: it treats substitutions as
    independent and therefore cannot represent epistasis. AssayPilot labels
    every multi-substitution score with ``additive_approximation=True``.
    """

    def __init__(
        self,
        reference: str,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        self.reference = reference
        self.checkpoint = checkpoint
        self.revision = revision
        self.device = device
        self._holder: _ESMModelHolder | None = None
        self._log_probs: dict[int, np.ndarray] = {}

    @property
    def holder(self) -> _ESMModelHolder:
        if self._holder is None:
            self._holder = _ESMModelHolder(self.checkpoint, self.revision, self.device)
        return self._holder

    def _compute_log_probs(self, positions: Sequence[int], batch_size: int = 8) -> None:
        """Fill the log-probability cache for the given 1-based positions."""
        needed = [p for p in sorted(set(positions)) if p not in self._log_probs]
        if not needed:
            return
        holder = self.holder
        torch = holder.torch
        tokenizer = holder.tokenizer

        for start in range(0, len(needed), batch_size):
            chunk = needed[start : start + batch_size]
            encoded = tokenizer([self.reference] * len(chunk), return_tensors="pt")
            input_ids = encoded["input_ids"].clone()
            for row, position in enumerate(chunk):
                input_ids[row, position] = tokenizer.mask_token_id  # +1 offset for <cls>
            encoded["input_ids"] = input_ids
            encoded = {k: v.to(holder.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = holder.model(**encoded).logits
            log_probs = torch.log_softmax(logits, dim=-1)

            for row, position in enumerate(chunk):
                row_probs = log_probs[row, position]
                self._log_probs[position] = np.array(
                    [float(row_probs[tokenizer.convert_tokens_to_ids(aa)]) for aa in AMINO_ACIDS],
                    dtype=np.float32,
                )

    def score(self, sequences: Sequence[str]) -> np.ndarray:
        """Score variants. Higher means more favoured by the pretrained model."""
        all_subs = [substitutions(self.reference, seq) for seq in sequences]
        positions = [s.position for subs in all_subs for s in subs]
        self._compute_log_probs(positions)

        aa_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        scores = np.zeros(len(sequences), dtype=np.float32)
        for row, subs in enumerate(all_subs):
            total = 0.0
            for sub in subs:
                table = self._log_probs[sub.position]
                total += float(table[aa_index[sub.mutant]] - table[aa_index[sub.wildtype]])
            scores[row] = total
        return scores

    def fingerprint(self) -> dict[str, Any]:
        resolved = self._holder.resolved_revision if self._holder is not None else self.revision
        return {
            "scorer": "esm2_masked_marginals",
            "checkpoint": self.checkpoint,
            "revision": resolved or "unpinned",
            "combination": "sum_of_single_site_scores",
            "additive_approximation": True,
        }
