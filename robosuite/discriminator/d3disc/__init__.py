"""D3-Disc: Dichotomous Density-ratio Discriminator.

Two-bank KNN failure detector built on top of the frozen flow_multi policy
encoder, with F3 soft-weight cleaning for the fail bank. Optional dynamics
predictor (``train_dynamics.py``) produces LPB-style 3-concat features.
"""

from .d3_benchmark import D3BenchmarkDiscriminator
from .dataset import LatentFlowDynamicsDataset
from .detector import D3Detector, DetectionResult
from .dynamics_feature import D3FeatureExtractor
from .encoder import EncodedDemo, FlowMultiEncoderWrapper
from .filter import compute_f3_weights, knn_sqdist, weighted_knn_sqdist
from .trainer import D3DynamicsTrainer, D3TrainerConfig

__all__ = [
    "D3BenchmarkDiscriminator",
    "D3Detector",
    "D3DynamicsTrainer",
    "D3FeatureExtractor",
    "D3TrainerConfig",
    "DetectionResult",
    "EncodedDemo",
    "FlowMultiEncoderWrapper",
    "LatentFlowDynamicsDataset",
    "compute_f3_weights",
    "knn_sqdist",
    "weighted_knn_sqdist",
]
