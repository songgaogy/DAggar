"""Generic core: source-agnostic FailureBenchmark, metrics, discriminator protocol."""

from .trajectory import BenchmarkTrajectory
from .discriminator import Discriminator, DiscriminatorOutput
from .benchmark import FailureBenchmark, BenchmarkResult, EvalConfig

__all__ = [
    "BenchmarkTrajectory",
    "Discriminator",
    "DiscriminatorOutput",
    "FailureBenchmark",
    "BenchmarkResult",
    "EvalConfig",
]
