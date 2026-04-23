"""D4-Disc: Bellman-Bootstrapped Dichotomous Discriminator.

Conditional latent dynamics critic with AdaLN-Zero + classifier-free-guidance
(CFG) scoring. See ``.claude/discriminator/d4_disc_design.md`` and
``.claude/discriminator/d4_implementation_plan.md`` for the design doc and
the module-by-module specification this package follows.
"""

from .adaln import (
    AdaLNDecoderBlock,
    AdaLNModulation,
    ConditionEmbedder,
)
from .d4_benchmark import D4BenchmarkDiscriminator
from .dataset import LatentFlowDynamicsDatasetD4
from .detector import D4Detector, D4StepOutput
from .dynamics_feature import D4FeatureExtractor, D4Frames
from .ema import ModelEMA
from .filter import compute_advantage_gate
from .model import ConditionalDynamicsPredictor
from .monitor import CollapseDetector, D4Health
from .schedule import D4Schedule
from .trainer import D4Trainer, D4TrainerConfig

__all__ = [
    "AdaLNDecoderBlock",
    "AdaLNModulation",
    "CollapseDetector",
    "ConditionEmbedder",
    "ConditionalDynamicsPredictor",
    "D4BenchmarkDiscriminator",
    "D4Detector",
    "D4FeatureExtractor",
    "D4Frames",
    "D4Health",
    "D4Schedule",
    "D4StepOutput",
    "D4Trainer",
    "D4TrainerConfig",
    "LatentFlowDynamicsDatasetD4",
    "ModelEMA",
    "compute_advantage_gate",
]
