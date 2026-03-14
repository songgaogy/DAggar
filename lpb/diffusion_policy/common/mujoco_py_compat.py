import os
import sys
import types


def install_mujoco_py_stub() -> bool:
    """
    Install a tiny `mujoco_py` stub to avoid importing native mujoco-py at runtime.

    robomimic's EnvRobosuite only needs `mujoco_py.builder.MujocoException`.
    In mixed Mujoco environments, importing real mujoco-py can cause native crashes
    or lock/build failures before rollout starts.

    Set LPB_MUJOCO_PY_STUB=0 to disable this behavior.
    """
    enabled = os.environ.get("LPB_MUJOCO_PY_STUB", "1").strip().lower()
    if enabled in {"0", "false", "no"}:
        return False

    if "mujoco_py" in sys.modules:
        return False

    module = types.ModuleType("mujoco_py")

    class MujocoException(Exception):
        pass

    builder = types.SimpleNamespace(MujocoException=MujocoException)
    module.builder = builder
    module.MujocoException = MujocoException

    sys.modules["mujoco_py"] = module
    return True

