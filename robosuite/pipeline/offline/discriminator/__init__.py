"""Offline discriminator finetuning components."""

from .checkpoint import (
    build_finetuned_checkpoint_payload,
    load_warmstart_detector,
    save_finetuned_checkpoint,
)
from .contracts import (
    optional_file,
    required_path,
    resolved_config_dict,
    safe_run_suffix,
    sha256_file,
    validate_finetune_contract,
)
from .encoder import FinetuneDynamicsEncoder
from .episodes import (
    GTNegativeWindow,
    PolicySegment,
    build_gt_negative_windows,
    load_offline_episodes,
    split_policy_segments,
    validate_offline_payload,
)
from .features import (
    build_action_windows,
    encode_gt_negative_windows,
    encode_policy_segments,
    exclude_gt_frames_from_unlabeled,
)
from .objectives import (
    CUDAPoolSampler,
    CompositeDiscriminatorObjective,
    LossTermConfig,
    NNPUParameters,
    ObjectiveResult,
    build_objective,
)
from .pools import (
    DiscriminatorPools,
    LatentTrajectory,
    feature_tensors,
    load_pretrain_pools,
    pool_stats,
)
from .trainer import (
    PUBCEDiscriminatorFT,
    finetune_warmstart_detector,
    require_cuda_device,
    resolve_parent_nnpu_semantics,
)

__all__ = [
    "DiscriminatorPools",
    "FinetuneDynamicsEncoder",
    "GTNegativeWindow",
    "LatentTrajectory",
    "LossTermConfig",
    "NNPUParameters",
    "ObjectiveResult",
    "PUBCEDiscriminatorFT",
    "PolicySegment",
    "CUDAPoolSampler",
    "CompositeDiscriminatorObjective",
    "build_action_windows",
    "build_gt_negative_windows",
    "build_objective",
    "build_finetuned_checkpoint_payload",
    "encode_gt_negative_windows",
    "encode_policy_segments",
    "exclude_gt_frames_from_unlabeled",
    "feature_tensors",
    "finetune_warmstart_detector",
    "load_offline_episodes",
    "load_pretrain_pools",
    "load_warmstart_detector",
    "optional_file",
    "pool_stats",
    "required_path",
    "require_cuda_device",
    "resolve_parent_nnpu_semantics",
    "resolved_config_dict",
    "safe_run_suffix",
    "save_finetuned_checkpoint",
    "sha256_file",
    "split_policy_segments",
    "validate_finetune_contract",
    "validate_offline_payload",
]
