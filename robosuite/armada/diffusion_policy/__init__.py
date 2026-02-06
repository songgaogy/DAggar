# robosuite/armada/diffusion_policy/__init__.py
# Compatibility layer: allow imports like `diffusion_policy.model` while the real code lives in
# `diffusion_policy.diffusion_policy.*`.

import importlib
import sys as _sys

_pkg = __name__  # "diffusion_policy"
_inner = f"{_pkg}.diffusion_policy"

# Subpackages that are commonly imported as `diffusion_policy.<name>`
_aliases = [
    "common",
    "codecs",
    "dataset",
    "env",
    "env_runner",
    "gym_util",
    "model",
    "policy",
    "scripts",
    "shared_memory",
    "workspace",
]

# Expose the inner top-level package as `diffusion_policy.diffusion_policy`
try:
    importlib.import_module(_inner)
except Exception:
    # If this fails, the package layout is different from expected.
    pass

for _name in _aliases:
    _src = f"{_inner}.{_name}"
    _dst = f"{_pkg}.{_name}"
    try:
        _mod = importlib.import_module(_src)
        _sys.modules[_dst] = _mod
        globals()[_name] = _mod
    except Exception:
        # Skip missing subpackages
        pass

del importlib, _sys, _pkg, _inner, _aliases, _name, _src, _dst, _mod
