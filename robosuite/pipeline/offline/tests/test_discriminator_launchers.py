"""Smoke tests for standalone discriminator launcher defaults."""

from __future__ import annotations

import os
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
        "append_optional_override SCHEDULER_HORIZON_EPOCHS "
        "offline.discriminator_finetune.scheduler_horizon_epochs"
        in finetune_text
    )
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
        "append_optional_override G_NORMALIZATION_ENABLED "
        "offline.discriminator_finetune.objective.logit_normalization.enabled"
        in finetune_text
    )
    assert (
        "append_optional_override QUADRATIC_CAP_ENABLED "
        "offline.discriminator_finetune.objective.quadratic_logit_cap.enabled"
        in finetune_text
    )
    assert (
        "append_optional_override QUADRATIC_CAP_C "
        "offline.discriminator_finetune.objective.quadratic_logit_cap.cap"
        in finetune_text
    )
    assert (
        "append_optional_override QUADRATIC_CAP_LAMBDA "
        "offline.discriminator_finetune.objective.quadratic_logit_cap.weight"
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
    assert 'GT_POSITIVE_TYPE:-positive_logistic' in finetune_text
    for config_key in (
        "safety_margin_weight",
        "margin_delta",
        "temperature",
        "boundary_source",
    ):
        assert (
            "+offline.discriminator_finetune.objective.terms.gt_positive."
            f"{config_key}="
        ) in finetune_text
    assert 'EPOCHS="${EPOCHS:-' not in finetune_text
    assert "NNPU_CKPT is required for warm-start Step 3 finetuning" in finetune_text
    assert "use_only_offline" not in finetune_text
    assert "pu_bce_eval_robosuite-chunk_v2" in finetune_text
    assert "discriminator-pretrain-quadratic-c2-l1e2-v2" in finetune_text

    visualize_text = visualize.read_text(encoding="utf-8")
    assert 'MODEL_CKPT="${MODEL_CKPT:-' in visualize_text
    assert 'NUM_TRAJS="${NUM_TRAJS:-10}"' in visualize_text
    assert "finetuned_scores_offline-success.pdf" in visualize_text
    assert "NUM_OFFLINE_SUCCESS_TRAJS" not in visualize_text

    evaluate_text = evaluate.read_text(encoding="utf-8")
    assert 'GENERATE_VISUALS="${GENERATE_VISUALS:-True}"' in evaluate_text
    assert 'SPLIT="${SPLIT:-both}"' in evaluate_text
    assert 'NUM_TRAJS="${NUM_TRAJS:-10}"' in evaluate_text
    assert (
        'VIS_OUT_DIR="${VIS_OUT_DIR:-${RUN_DIR}/visualization/val-seed${SEED}}"'
        in evaluate_text
    )
    assert "robosuite/pipeline/offline/scripts/vis_disc_finetuned.sh" in evaluate_text
    assert 'OUT_DIR="${VIS_OUT_DIR}"' in evaluate_text
    assert "gt_fail_detection.json" in evaluate_text
    assert "finetuned_scores_offline-success.pdf" in evaluate_text
    assert "visualization bundle disabled" in evaluate_text
    assert evaluate_text.count('"$@"') == 1


def test_eval_launcher_keeps_evaluation_and_visualization_outputs_separate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "pu_bce_head_finetuned.pth"
    model_checkpoint = tmp_path / "model_10.pth"
    offline_episodes = tmp_path / "offline_episodes.pt"
    for path in (checkpoint, model_checkpoint, offline_episodes):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    invocation_log = tmp_path / "python_invocations.txt"
    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "${INVOCATION_LOG}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "ROOT_DIR": str(REPO_ROOT),
            "PY": str(fake_python),
            "INVOCATION_LOG": str(invocation_log),
            "TASK": "PickPlaceCereal",
            "FINETUNED_CKPT": str(checkpoint),
            "MODEL_CKPT": str(model_checkpoint),
            "OFFLINE_EPISODES": str(offline_episodes),
            "DEVICE": "cuda:0",
        }
    )
    subprocess.run(
        [
            "bash",
            str(SCRIPTS / "eval_disc_finetuned.sh"),
            "--eval-only-token",
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    evaluation_dir = run_dir / "evaluation" / "val-seed0"
    visualization_dir = run_dir / "visualization" / "val-seed0"
    assert evaluation_dir.is_dir()
    assert visualization_dir.is_dir()
    assert evaluation_dir != visualization_dir

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 2
    assert "robosuite.pipeline.offline.src.eval_disc_checkpoint" in invocations[0]
    assert f"--out-dir {evaluation_dir}" in invocations[0]
    assert "--eval-only-token" in invocations[0]
    assert "robosuite.pipeline.offline.src.visualize_disc_finetuned" in invocations[1]
    assert f"--out-dir {visualization_dir}" in invocations[1]
    assert "--eval-only-token" not in invocations[1]
    assert "robosuite.pipeline.offline.discriminator.test_finetuned" not in (
        SCRIPTS / "vis_disc_finetuned.sh"
    ).read_text(encoding="utf-8")


def test_eval_without_visuals_still_owns_gt_fail_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "pu_bce_head_finetuned.pth"
    model_checkpoint = tmp_path / "model_10.pth"
    offline_episodes = tmp_path / "offline_episodes.pt"
    for path in (checkpoint, model_checkpoint, offline_episodes):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    invocation_log = tmp_path / "python_invocations.txt"
    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "${INVOCATION_LOG}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "ROOT_DIR": str(REPO_ROOT),
            "PY": str(fake_python),
            "INVOCATION_LOG": str(invocation_log),
            "FINETUNED_CKPT": str(checkpoint),
            "MODEL_CKPT": str(model_checkpoint),
            "OFFLINE_EPISODES": str(offline_episodes),
            "GENERATE_VISUALS": "False",
        }
    )

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "eval_disc_finetuned.sh")],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "robosuite.pipeline.offline.src.eval_disc_checkpoint" in invocations[0]
    assert "gt_fail_detection.json" in completed.stdout


def test_one_off_tuning_entrypoints_are_removed() -> None:
    assert not (SCRIPTS / "tune_disc_finetune.sh").exists()
    assert not (
        REPO_ROOT
        / "robosuite"
        / "pipeline"
        / "offline"
        / "src"
        / "summarize_disc_tuning.py"
    ).exists()


def test_finetune_config_defaults_to_three_independent_risks() -> None:
    cfg = OmegaConf.load(CONFIG)
    finetune = cfg.offline.discriminator_finetune

    assert "use_only_offline" not in finetune
    assert "pu_bce_eval_robosuite-chunk_v2" in cfg.algorithm.discriminator.checkpoint
    assert finetune.pretrain_dir.endswith(
        "discriminator-pretrain-quadratic-c2-l1e2-v2"
    )
    assert finetune.epochs == 10
    assert finetune.scheduler_horizon_epochs == 20
    assert finetune.lr == 1.0e-5
    assert finetune.gt_negative.pre_intervention_chunks == 1
    assert finetune.gt_negative.post_intervention_chunks == 1
    assert finetune.gt_negative.pre_end_chunk == 0
    assert finetune.gt_negative.action_source == "policy_action"
    assert finetune.objective.steps_per_epoch is None
    normalization = finetune.objective.logit_normalization
    assert normalization.enabled is True
    assert normalization.method == "fixed_robust_iqr"
    quadratic_cap = finetune.objective.quadratic_logit_cap
    assert quadratic_cap.enabled is True
    assert quadratic_cap.scope == "nnpu_replay"
    assert quadratic_cap.cap == 2.0
    assert quadratic_cap.weight == 1.0e-2
    assert finetune.objective.terms.nnpu_replay.batch_size == 512
    assert finetune.objective.terms.nnpu_replay.weight == 0.5
    assert finetune.objective.terms.nnpu_replay.positive_fraction == 0.5
    gt_positive = finetune.objective.terms.gt_positive
    assert gt_positive.type == "positive_logistic"
    assert gt_positive.weight == 0.1
    assert gt_positive.batch_size == 256
    gt_negative = finetune.objective.terms.gt_negative
    assert gt_negative.type == "negative_logistic"
    assert gt_negative.weight == 0.00625
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
