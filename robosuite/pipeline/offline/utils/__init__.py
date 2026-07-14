"""Tool functions for offline DIPOLE training."""

from .advantage import (
    OfflineAdvantageGProvider,
    VASTAdvantageDiagnostics,
    precompute_offline_advantage,
    precompute_vast_offline_advantage,
    sample_vast_macro_horizons,
)
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
from .vast_finetune import (
    build_vast_finetune_buffer,
    finetune_vast,
    save_finetuned_vast,
    validate_vast_checkpoint_payload,
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
    "VASTAdvantageDiagnostics",
    "RoutedSigmoidBranchWeightPolicy",
    "build_agent_env",
    "build_branch_weight_policy",
    "build_vast_finetune_buffer",
    "build_offline_transitions",
    "build_online_success_transitions",
    "finalize_normalizers",
    "finetune_vast",
    "load_offline_data_transitions",
    "load_pretrain_transitions",
    "make_hdf5_loader",
    "populate_replay_buffer",
    "precompute_neg_all_membership",
    "precompute_offline_advantage",
    "precompute_vast_offline_advantage",
    "sample_vast_macro_horizons",
    "print_policy_param_summary",
    "save_finetuned_vast",
    "validate_vast_checkpoint_payload",
]
