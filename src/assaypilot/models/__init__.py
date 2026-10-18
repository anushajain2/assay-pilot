"""The fixed comparison suite: baseline, pretrained score, assay-trained model."""

from .assay_trained import EmbeddingRidgeMethod
from .base import Method, MethodInfo
from .baseline import OneHotRidgeBaseline
from .pretrained import PretrainedScoreMethod
from .registry import SuiteConfig, build_backend, build_cache, build_methods, split_by_availability

__all__ = [
    "EmbeddingRidgeMethod",
    "Method",
    "MethodInfo",
    "OneHotRidgeBaseline",
    "PretrainedScoreMethod",
    "SuiteConfig",
    "build_backend",
    "build_cache",
    "build_methods",
    "split_by_availability",
]
