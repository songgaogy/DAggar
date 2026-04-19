"""LPB score: latent trajectory DSM training and offline failure detection.

Pipeline (training):
    Raw demos (HDF5) -> split/refs per task -> optional preprocessed cache ->
    FlowMultitaskEncoder (frozen policy vision + language) maps windows to latents ->
    DSMModel adds noise on normalized latents; ChunkConditionedDSM denoises under
    task + trajectory-type + sigma -> DSM + contrastive loss (Trainer).

Inference (see ``core/dsm_discriminator``):
    Same encoder + trained DSM; score chunks via positive vs failure branches and
    T3-style detector statistics.

Public symbols below are re-exported for library use; CLI entrypoints are
``train.py`` and ``visualize_failures.py``.
"""

from .analysis import (
    TERM_COLORS,
    TERM_LABELS,
    TERM_ORDER,
    TERM_SHORT_LABELS,
    map_term_values_to_frames,
    ordered_term_keys,
    summarize_result_set_term_attribution,
    summarize_trajectory_term_attribution,
)
from .app import (
    TrainingDatasets,
    build_dsm_discriminator,
    build_flow_encoder,
    build_training_datasets,
    load_split_trajectories,
    now_tag,
    run_train,
    run_visualize,
    select_split_refs,
)
from .core import (
    DEFAULT_TASK_ORDER,
    ConditionalManifoldDenoiser,
    DATA_TYPE_ORDER,
    DemoRef,
    DSMDiscriminator,
    DSMModel,
    DSMTransitionScorer,
    FlowMultitaskEncoder,
    MODEL_ARCHITECTURE,
    PreparedTrajectory,
    JointManifoldDenoiser,
    LatentTransitionDataset,
    SplitCounts,
    TaskDataSpec,
    Trainer,
    TrainerConfig,
    TrajectoryScoreBundle,
    UnifiedConditionedDSM,
    build_conditional_manifold_denoiser,
    build_dsm_model,
    build_joint_manifold_denoiser,
    build_split_refs,
    build_unified_conditioned_dsm,
    estimate_trajectories_nbytes,
    filter_refs_by_data_types,
    normalize_task_name,
    ordered_task_names,
    prepare_trajectories,
    resolve_checkpoint_task_name,
    resolve_window_size,
)

__all__ = [
    "TERM_ORDER",
    "TERM_LABELS",
    "TERM_SHORT_LABELS",
    "TERM_COLORS",
    "ordered_term_keys",
    "map_term_values_to_frames",
    "summarize_trajectory_term_attribution",
    "summarize_result_set_term_attribution",
    "ConditionalManifoldDenoiser",
    "UnifiedConditionedDSM",
    "DATA_TYPE_ORDER",
    "DEFAULT_TASK_ORDER",
    "SplitCounts",
    "TaskDataSpec",
    "DemoRef",
    "PreparedTrajectory",
    "LatentTransitionDataset",
    "FlowMultitaskEncoder",
    "MODEL_ARCHITECTURE",
    "normalize_task_name",
    "resolve_checkpoint_task_name",
    "ordered_task_names",
    "build_split_refs",
    "filter_refs_by_data_types",
    "prepare_trajectories",
    "estimate_trajectories_nbytes",
    "resolve_window_size",
    "JointManifoldDenoiser",
    "DSMModel",
    "build_conditional_manifold_denoiser",
    "build_unified_conditioned_dsm",
    "build_joint_manifold_denoiser",
    "build_dsm_model",
    "TrainerConfig",
    "Trainer",
    "DSMTransitionScorer",
    "DSMDiscriminator",
    "TrajectoryScoreBundle",
    "TrainingDatasets",
    "now_tag",
    "build_flow_encoder",
    "build_dsm_discriminator",
    "select_split_refs",
    "load_split_trajectories",
    "build_training_datasets",
    "run_train",
    "run_visualize",
]
