"""Tool functions for offline DIPOLE training."""

from .advantage import OfflineAdvantageGProvider, precompute_offline_advantage
from .branch_weights import (
    BranchWeightPolicy,
    DiscriminatorScaledBranchWeightPolicy,
    RoutedSigmoidBranchWeightPolicy,
    build_branch_weight_policy,
)
from .buffer import (
    load_offline_data_transitions,
    load_pretrain_transitions,
    populate_replay_buffer,
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
from .iql_finetune import (
    build_iql_finetune_buffer,
    finetune_iql,
    save_finetuned_iql,
)
from .setup import (
    AgentEnvContext,
    build_agent_env,
    finalize_normalizers,
    make_hdf5_loader,
    print_policy_param_summary,
)

__all__ = [
    "AgentEnvContext",
    "BranchWeightPolicy",
    "DiscriminatorScaledBranchWeightPolicy",
    "NaiveNegativeGProvider",
    "NegAllGProvider",
    "OfflineAdvantageGProvider",
    "OfflineStreams",
    "RoutedSigmoidBranchWeightPolicy",
    "build_agent_env",
    "build_branch_weight_policy",
    "build_iql_finetune_buffer",
    "build_offline_transitions",
    "build_online_success_transitions",
    "finalize_normalizers",
    "finetune_iql",
    "load_offline_data_transitions",
    "load_pretrain_transitions",
    "make_hdf5_loader",
    "populate_replay_buffer",
    "precompute_neg_all_membership",
    "precompute_offline_advantage",
    "print_policy_param_summary",
    "save_finetuned_iql",
]
