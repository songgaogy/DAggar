"""Tool functions for offline DIPOLE training."""

from .branch_weights import (
    BranchWeightPolicy,
    DiscriminatorScaledBranchWeightPolicy,
    RoutedSigmoidBranchWeightPolicy,
    build_branch_weight_policy,
)
from .buffer import (
    load_pretrain_transitions,
    populate_replay_buffer,
)
from .discriminator_scores import (
    OfflineDiscriminatorGProvider,
    precompute_discriminator_scores,
)
from .episode_dataset import (
    OfflineStreams,
    build_offline_transitions,
    build_online_success_transitions,
)
from .hard_label_providers import (
    NaiveNegativeGProvider,
    NegAllGProvider,
    precompute_neg_all_membership,
)
from .setup import (
    AgentEnvContext,
    build_agent_env,
    finalize_normalizers,
    make_hdf5_loader,
    print_policy_param_summary,
)
from .policy_training import (
    POLICY_TRAINING_COUPLED,
    POLICY_TRAINING_FILTERED_BC,
    POLICY_TRAINING_INDEPENDENT_SOFT,
    branch_only_update,
    branch_seed,
    normalize_policy_training_mode,
    parameter_distance_metrics,
    sample_static_cache,
    trainable_parameter_snapshot,
)
from .provenance import (
    directory_input_provenance,
    file_provenance,
    git_provenance,
    manifest_shard_provenance,
)

__all__ = [
    "AgentEnvContext",
    "BranchWeightPolicy",
    "DiscriminatorScaledBranchWeightPolicy",
    "NaiveNegativeGProvider",
    "NegAllGProvider",
    "OfflineDiscriminatorGProvider",
    "OfflineStreams",
    "RoutedSigmoidBranchWeightPolicy",
    "POLICY_TRAINING_COUPLED",
    "POLICY_TRAINING_FILTERED_BC",
    "POLICY_TRAINING_INDEPENDENT_SOFT",
    "branch_only_update",
    "branch_seed",
    "build_agent_env",
    "build_branch_weight_policy",
    "build_offline_transitions",
    "build_online_success_transitions",
    "finalize_normalizers",
    "directory_input_provenance",
    "file_provenance",
    "git_provenance",
    "load_pretrain_transitions",
    "make_hdf5_loader",
    "manifest_shard_provenance",
    "normalize_policy_training_mode",
    "parameter_distance_metrics",
    "populate_replay_buffer",
    "precompute_neg_all_membership",
    "precompute_discriminator_scores",
    "print_policy_param_summary",
    "sample_static_cache",
    "trainable_parameter_snapshot",
]
