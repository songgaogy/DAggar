"""Public API for the dyn_disc PU-BCE discriminator.

Imports are lazy (PEP 562): the ``adapters`` symbols pull in the optional
``benchmark`` package, which is only needed for offline benchmark evaluation. Online
consumers (e.g. the Flow-DAgger runtime) use only the encoder/detector symbols and
must be able to import them without ``benchmark`` installed. Accessing an adapter
symbol still resolves it on demand and will raise if ``benchmark`` is unavailable.
"""

from typing import TYPE_CHECKING

# Map each public name to its defining submodule for lazy resolution.
_LAZY = {
    "PUBCEBenchmarkDiscriminator": ".adapters.pu_bce",
    "DynBenchmarkDiscriminator": ".adapters.single_bank",
    "load_model": ".core.model_loader",
    "BCEHead": ".detectors.pu_bce",
    "PUBCEDiscriminator": ".detectors.pu_bce",
    "PUCalibStats": ".detectors.pu_bce",
    "pu_risk": ".detectors.pu_bce",
    "DetectionResult": ".detectors.single_bank_knn",
    "DynEncoder": ".detectors.single_bank_knn",
    "knn_min_l2_dist": ".detectors.single_bank_knn",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):  # PEP 562 module-level lazy attribute access
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # for static analyzers / IDEs only
    from .adapters.pu_bce import PUBCEBenchmarkDiscriminator
    from .adapters.single_bank import DynBenchmarkDiscriminator
    from .core.model_loader import load_model
    from .detectors.pu_bce import BCEHead, PUBCEDiscriminator, PUCalibStats, pu_risk
    from .detectors.single_bank_knn import DetectionResult, DynEncoder, knn_min_l2_dist
