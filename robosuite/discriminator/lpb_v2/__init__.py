from .core.model_loader import load_model
from .detectors.bce import BCECalibStats, BCEDiscriminator, BCEHead
from .detectors.single_bank_knn import DetectionResult, LPBV2Encoder, LPBV2KNN, knn_min_l2_dist
from .detectors.two_bank_knn import TwoBankKNN

__all__ = [
    "BCEBenchmarkDiscriminator",
    "BCECalibStats",
    "BCEDiscriminator",
    "BCEHead",
    "DetectionResult",
    "LPBV2BenchmarkDiscriminator",
    "LPBV2Encoder",
    "LPBV2KNN",
    "TwoBankBenchmarkDiscriminator",
    "TwoBankKNN",
    "knn_min_l2_dist",
    "load_model",
]

_LAZY_IMPORTS = {
    "BCEBenchmarkDiscriminator": (".adapters.bce", "BCEBenchmarkDiscriminator"),
    "LPBV2BenchmarkDiscriminator": (".adapters.single_bank", "LPBV2BenchmarkDiscriminator"),
    "TwoBankBenchmarkDiscriminator": (".adapters.two_bank", "TwoBankBenchmarkDiscriminator"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
