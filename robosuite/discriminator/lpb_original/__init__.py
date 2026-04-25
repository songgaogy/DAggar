"""Original LPB KNN OOD discriminator (faithful port from tmp/lpb-main).

Layout:
    dyn_model/                   verbatim copy of upstream tmp/lpb-main/dyn_model
    _vendored/diffusion_policy/  pruned copy (env/ & online-eval subdirs removed)
    datasets/                    HDF5 dataset adapter to feed user's data
    discriminator.py             KNN OOD scoring class
    lpb_benchmark.py             adapter to BenchmarkTrajectory + DiscriminatorOutput
    train.py                     dynamics training entry (Hydra)
    conf/                        copied Hydra configs

Importing this package injects vendored paths into sys.path so upstream
absolute imports (`import dyn_model`, `from diffusion_policy....`) resolve
to the vendored copies without source rewrites.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENDORED = _HERE / "_vendored"

# Make `import dyn_model` resolve to lpb_original/dyn_model
_pkg_dir = str(_HERE)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Make `from diffusion_policy....` resolve to the vendored copy
_vendored_dir = str(_VENDORED)
if _vendored_dir not in sys.path:
    sys.path.insert(0, _vendored_dir)
