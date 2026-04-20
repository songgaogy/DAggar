"""logpZO (FAIL-Detect) failure-detection module.

Implements the logpZO variant from arXiv:2503.08558:
    * Stage 1: normalizing-flow density estimation of success-state occupancy.
    * Stage 2: Conformal-Prediction threshold calibration on success data.
"""

from .flow_model import RealNVPFlow
from .logpZO_benchmark import LogpZOBenchmarkDiscriminator
from .monitor import FAILDetectMonitor, FitStats

__all__ = [
    "RealNVPFlow",
    "FAILDetectMonitor",
    "FitStats",
    "LogpZOBenchmarkDiscriminator",
]
