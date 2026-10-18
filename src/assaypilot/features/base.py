"""Representation backends and the embedding cache.

A *backend* turns sequences into a fixed-size numeric matrix. Two are shipped:

``esm2``
    Frozen ESM-2 embeddings (optional install: ``pip install assaypilot[esm]``).
``physchem``
    A dependency-free physicochemical representation. It exists so the whole
    workflow — tests, demos, CI — runs without downloading model weights. It is
    **not** a protein language model and is labelled as such everywhere it
    appears.

Every backend exposes a :meth:`EmbeddingBackend.fingerprint`, which identifies
the representation precisely enough to reproduce it: name, checkpoint, pinned
revision, pooling, layer, and output dimension. The fingerprint is both the
cache key and a field in the run manifest, because a model *name* alone is not
enough to reproduce a result.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..sequences import sequence_hash


def default_cache_dir() -> Path:
    """Where embeddings are cached unless the caller says otherwise.

    Honours ``ASSAYPILOT_CACHE_DIR``, then ``XDG_CACHE_HOME``, then
    ``~/.cache/assaypilot``.
    """
    env = os.environ.get("ASSAYPILOT_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "assaypilot" / "embeddings"


class EmbeddingBackend(ABC):
    """Turns sequences into a matrix of shape ``(n_sequences, n_features)``."""

    #: Short identifier used in reports, for example ``"esm2"``.
    name: str = "backend"

    @abstractmethod
    def embed(self, sequences: Sequence[str]) -> np.ndarray:
        """Embed sequences in the given order. Implementations need no caching."""

    @abstractmethod
    def fingerprint(self) -> dict[str, Any]:
        """Everything needed to reproduce this representation exactly."""

    @property
    def is_language_model(self) -> bool:
        """Whether this representation comes from a pretrained protein language model."""
        return False

    @property
    def description(self) -> str:
        """One line shown to the researcher in reports."""
        return self.name

    def fingerprint_key(self) -> str:
        """Stable short hash of the fingerprint, used as the cache namespace."""
        blob = json.dumps(self.fingerprint(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


class EmbeddingCache:
    """Disk cache keyed by sequence hash and backend fingerprint.

    Entries live at ``<root>/<backend-key>/<first two hash chars>/<hash>.npy``.
    A cache hit requires both the same sequence and the same representation
    configuration, so switching checkpoints can never reuse the wrong vectors.
    """

    def __init__(self, root: str | Path | None = None, *, enabled: bool = True) -> None:
        self.root = Path(root) if root is not None else default_cache_dir()
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, namespace: str, seq_hash: str) -> Path:
        return self.root / namespace / seq_hash[:2] / f"{seq_hash}.npy"

    def get(self, namespace: str, seq_hash: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        path = self._path(namespace, seq_hash)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            # A truncated or unreadable cache entry is treated as a miss, never
            # as a failure of the run.
            return None

    def put(self, namespace: str, seq_hash: str, vector: np.ndarray) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, seq_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a process-specific temporary file and rename, so a cache
        # entry is never half-written and concurrent runs cannot collide.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as handle:
            np.save(handle, vector)
        tmp.replace(path)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "root": str(self.root),
            "hits": self.hits,
            "misses": self.misses,
        }


def embed_sequences(
    backend: EmbeddingBackend,
    sequences: Iterable[str],
    *,
    cache: EmbeddingCache | None = None,
    batch_size: int = 16,
) -> np.ndarray:
    """Embed sequences through ``backend``, reusing cached vectors where possible.

    Duplicate sequences are embedded once. The returned matrix follows the input
    order, including duplicates.
    """
    sequences = list(sequences)
    if not sequences:
        return np.zeros((0, 0), dtype=np.float32)

    namespace = backend.fingerprint_key()
    hashes = [sequence_hash(s) for s in sequences]
    vectors: dict[str, np.ndarray] = {}
    missing: list[str] = []

    for seq, seq_hash in zip(sequences, hashes, strict=True):
        if seq_hash in vectors:
            continue
        cached = cache.get(namespace, seq_hash) if cache is not None else None
        if cached is not None:
            vectors[seq_hash] = cached
            if cache is not None:
                cache.hits += 1
        elif seq not in missing:
            missing.append(seq)

    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        computed = backend.embed(chunk)
        for seq, vector in zip(chunk, computed, strict=True):
            seq_hash = sequence_hash(seq)
            vector = np.asarray(vector, dtype=np.float32)
            vectors[seq_hash] = vector
            if cache is not None:
                cache.put(namespace, seq_hash, vector)
                cache.misses += 1

    return np.vstack([vectors[h] for h in hashes]).astype(np.float32)
