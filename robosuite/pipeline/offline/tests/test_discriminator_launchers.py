"""Smoke tests for standalone discriminator launcher defaults."""

from __future__ import annotations

import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from robosuite.pipeline.offline.discriminator.finetune_setup import (
    configured_loss_terms,
    resolved_sampler_config,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "robosuite" / "pipeline" / "offline" / "scripts"
CONFIG = REPO_ROOT / "robosuite" / "pipeline" / "config" / "finetune_disc.yaml"


def test_discriminator_launchers_parse_with_safe_environment_defaults() -> None:
    finetune = SCRIPTS / "finetune_disc.sh"
    visualize = SCRIPTS / "vis_disc_finetuned.sh"
    evaluate = SCRIPTS / "eval_disc_finetuned.sh"

    for script in (finetune, visualize, evaluate):
        subprocess.run(["bash", "-n", str(script)], check=True)

    finetune_text = finetune.read_text(encoding="utf-8")
    assert 'NNPU_ENCODER_CKPT="${NNPU_ENCODER_CKPT:-' in finetune_text
    assert 'NNPU_CAMERA_TO_VIEW="${NNPU_CAMERA_TO_VIEW:-}"' in finetune_text
    assert "append_optional_override EPOCHS offline.discriminator_finetune.epochs" in finetune_text
    assert (
        "append_optional_override LAMBDA_PRE "
        "offline.discriminator_finetune.objective.terms.nnpu_replay.weight"
        in finetune_text
    )
    assert (
        "append_optional_override LAMBDA_P "
        "offline.discriminator_finetune.objective.terms.gt_positive.weight"
        in finetune_text
    )
    assert (
        "append_optional_override LAMBDA_N "
        "offline.discriminator_finetune.objective.terms.gt_negative.weight"
        in finetune_text
    )
    assert (
        "append_optional_override GT_POSITIVE_BATCH_SIZE "
        "offline.discriminator_finetune.objective.terms.gt_positive.batch_size"
        in finetune_text
    )
    assert (
        "append_optional_override GT_NEGATIVE_BATCH_SIZE "
        "offline.discriminator_finetune.objective.terms.gt_negative.batch_size"
        in finetune_text
    )
    for env_name, config_key in (
        ("SAFETY_MARGIN_WEIGHT", "safety_margin_weight"),
        ("SAFETY_MARGIN_DELTA", "margin_delta"),
        ("SAFETY_MARGIN_TEMPERATURE", "temperature"),
        ("SAFETY_MARGIN_BOUNDARY_SOURCE", "boundary_source"),
    ):
        assert (
            f"append_optional_override {env_name} "
            "offline.discriminator_finetune.objective.terms.gt_positive."
            f"{config_key}"
        ) in finetune_text
    assert 'EPOCHS="${EPOCHS:-' not in finetune_text
    assert "NNPU_CKPT is required for warm-start Step 3 finetuning" in finetune_text
    assert "use_only_offline" not in finetune_text

    visualize_text = visualize.read_text(encoding="utf-8")
    assert 'MODEL_CKPT="${MODEL_CKPT:-' in visualize_text
    assert 'NUM_TRAJS="${NUM_TRAJS:-10}"' in visualize_text
    assert "finetuned_scores_offline-success.pdf" in visualize_text
    assert "NUM_OFFLINE_SUCCESS_TRAJS" not in visualize_text


def test_finetune_config_defaults_to_three_independent_risks() -> None:
    cfg = OmegaConf.load(CONFIG)
    finetune = cfg.offline.discriminator_finetune

    assert "use_only_offline" not in finetune
    assert finetune.epochs == 10
    assert finetune.lr == 3.0e-5
    assert finetune.gt_negative.pre_intervention_chunks == 1
    assert finetune.gt_negative.post_intervention_chunks == 1
    assert finetune.gt_negative.pre_end_chunk == 0
    assert finetune.gt_negative.action_source == "policy_action"
    assert finetune.objective.steps_per_epoch is None
    assert finetune.objective.terms.nnpu_replay.batch_size == 512
    assert finetune.objective.terms.nnpu_replay.positive_fraction == 0.5
    gt_positive = finetune.objective.terms.gt_positive
    assert gt_positive.type == "positive_safety_margin"
    assert gt_positive.weight == 0.025
    assert gt_positive.batch_size == 256
    assert gt_positive.safety_margin_weight == 1.0
    assert gt_positive.margin_delta == 1.0
    assert gt_positive.temperature == 1.0
    assert gt_positive.boundary_source == "parent_checkpoint"
    gt_negative = finetune.objective.terms.gt_negative
    assert gt_negative.type == "negative_logistic"
    assert gt_negative.weight == 0.0125
    assert gt_negative.batch_size == 256

    objective = OmegaConf.to_container(finetune.objective, resolve=True)
    assert isinstance(objective, dict)
    sampler = resolved_sampler_config(
        configured_loss_terms(objective),
        seed=int(cfg.seed),
        device=str(cfg.algorithm.discriminator.learner_device),
    )
    assert sampler["strategy"] == "independent_uniform_with_replacement"
    assert sampler["terms"]["nnpu_replay"]["pool_batch_sizes"] == {
        "pretrain_positive": 256,
        "pretrain_unlabeled": 256,
    }
    assert sampler["terms"]["gt_positive"]["pool_batch_sizes"] == {
        "offline_positive": 256,
    }
    assert sampler["terms"]["gt_negative"]["pool_batch_sizes"] == {
        "offline_gt_negative": 256,
    }
