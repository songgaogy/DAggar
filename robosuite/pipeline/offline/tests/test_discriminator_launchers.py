"""Behavior tests for discriminator and offline pipeline launchers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from robosuite.pipeline.offline.discriminator.finetune_setup import (
    configured_loss_terms,
    resolved_sampler_config,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "robosuite" / "pipeline" / "offline" / "scripts"
CONFIG_DIR = REPO_ROOT / "robosuite" / "pipeline" / "config"
CONFIG = CONFIG_DIR / "discriminator.yaml"


def _compose_config(config_name: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def _fake_python(tmp_path: Path) -> tuple[Path, Path]:
    invocation_log = tmp_path / "python_invocations.txt"
    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "${INVOCATION_LOG}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python, invocation_log


def test_discriminator_launchers_parse_with_safe_environment_defaults() -> None:
    finetune = SCRIPTS / "finetune_disc.sh"
    evaluate = SCRIPTS / "eval_disc_finetuned.sh"
    train = SCRIPTS / "train_offline_dipole.sh"

    for script in (finetune, evaluate, train):
        subprocess.run(["bash", "-n", str(script)], check=True)

    evaluate_text = evaluate.read_text(encoding="utf-8")
    assert 'DELTA="${DELTA:-5.0}"' in evaluate_text
    assert evaluate_text.count('--delta "${DELTA}"') == 2


def test_finetune_launcher_creates_discriminator_stage(tmp_path: Path) -> None:
    parent_checkpoint = tmp_path / "pu_bce_head.pth"
    encoder_checkpoint = tmp_path / "model_10.pth"
    offline_episodes = tmp_path / "offline_episodes.pt"
    pretrain_dir = tmp_path / "pretrain"
    for path in (parent_checkpoint, encoder_checkpoint, offline_episodes):
        path.touch()
    pretrain_dir.mkdir()
    (pretrain_dir / "manifest.json").write_text("{}", encoding="utf-8")
    fake_python, invocation_log = _fake_python(tmp_path)
    run_root = tmp_path / "runs"
    environment = os.environ.copy()
    environment.update(
        {
            "ROOT_DIR": str(REPO_ROOT),
            "PY": str(fake_python),
            "INVOCATION_LOG": str(invocation_log),
            "TASK": "PickPlaceCereal",
            "NNPU_CKPT": str(parent_checkpoint),
            "NNPU_ENCODER_CKPT": str(encoder_checkpoint),
            "OFFLINE_EPISODES": str(offline_episodes),
            "PRETRAIN_DIR": str(pretrain_dir),
            "RUN_ROOT": str(run_root),
            "RUN_SUBFIX": "contract",
        }
    )

    completed = subprocess.run(
        ["bash", str(SCRIPTS / "finetune_disc.sh")],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    pipeline_line = next(
        line for line in completed.stdout.splitlines() if "pipeline_run_dir=" in line
    )
    pipeline_dir = Path(pipeline_line.split("=", 1)[1])
    assert pipeline_dir.parent == run_root.resolve()
    assert pipeline_dir.name.endswith("_contract")
    assert pipeline_dir.is_dir()
    invocation = invocation_log.read_text(encoding="utf-8")
    assert (
        f"offline.discriminator_finetune.run_dir={pipeline_dir}/discriminator"
        in invocation
    )


def test_train_launcher_requires_pipeline_and_uses_finetuned_checkpoint(
    tmp_path: Path,
) -> None:
    pipeline_dir = tmp_path / "PickPlaceCereal_run"
    checkpoint = (
        pipeline_dir
        / "discriminator"
        / "checkpoints"
        / "pu_bce_head_finetuned.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    fake_python, invocation_log = _fake_python(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "ROOT_DIR": str(REPO_ROOT),
            "PY": str(fake_python),
            "INVOCATION_LOG": str(invocation_log),
            "PIPELINE_RUN_DIR": str(pipeline_dir),
        }
    )

    subprocess.run(
        ["bash", str(SCRIPTS / "train_offline_dipole.sh")],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    invocation = invocation_log.read_text(encoding="utf-8")
    assert f"algorithm.discriminator.checkpoint={checkpoint}" in invocation
    assert f"offline.run_dir={pipeline_dir}/dipole" in invocation

    environment.pop("PIPELINE_RUN_DIR")
    missing = subprocess.run(
        ["bash", str(SCRIPTS / "train_offline_dipole.sh")],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert missing.returncode != 0
    assert "PIPELINE_RUN_DIR is required" in missing.stderr


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


def test_eval_launcher_supports_visualization_only(tmp_path: Path) -> None:
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
            "RUN_EVAL": "False",
        }
    )

    subprocess.run(
        ["bash", str(SCRIPTS / "eval_disc_finetuned.sh"), "--eval-only-token"],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    visualization_dir = run_dir / "visualization" / "val-seed0"
    assert visualization_dir.is_dir()
    assert not (run_dir / "evaluation" / "val-seed0").exists()

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "robosuite.pipeline.offline.src.visualize_disc_finetuned" in invocations[0]
    assert f"--out-dir {visualization_dir}" in invocations[0]
    assert "--eval-only-token" not in invocations[0]

def test_finetune_config_defaults_to_three_independent_risks() -> None:
    cfg = _compose_config("discriminator")
    finetune = cfg.offline.discriminator_finetune

    assert "use_only_offline" not in finetune
    assert cfg.algorithm.discriminator.checkpoint is None
    assert "pu_bce_eval_robosuite-chunk_v2" in finetune.parent_checkpoint
    assert finetune.pretrain_dir.endswith(
        "discriminator-pretrain-quadratic-c2-l1e2-v2"
    )
    assert finetune.epochs == 20
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
    assert quadratic_cap.scope == "all_terms"
    assert quadratic_cap.cap == 2.0
    assert quadratic_cap.weight == 1.0e-2
    assert finetune.objective.terms.nnpu_replay.batch_size == 256
    assert finetune.objective.terms.nnpu_replay.weight == 0.1
    assert finetune.objective.terms.nnpu_replay.positive_fraction == 0.5
    gt_positive = finetune.objective.terms.gt_positive
    assert gt_positive.type == "positive_logistic"
    assert gt_positive.weight == 0.2
    assert gt_positive.batch_size == 128
    gt_negative = finetune.objective.terms.gt_negative
    assert gt_negative.type == "negative_logistic"
    assert gt_negative.weight == 0.01
    assert gt_negative.batch_size == 128

    objective = OmegaConf.to_container(finetune.objective, resolve=True)
    assert isinstance(objective, dict)
    sampler = resolved_sampler_config(
        configured_loss_terms(objective),
        seed=int(cfg.seed),
        device=str(cfg.algorithm.discriminator.learner_device),
    )
    assert sampler["strategy"] == "independent_uniform_with_replacement"
    assert sampler["terms"]["nnpu_replay"]["pool_batch_sizes"] == {
        "pretrain_positive": 128,
        "pretrain_unlabeled": 128,
    }
    assert sampler["terms"]["gt_positive"]["pool_batch_sizes"] == {
        "offline_positive": 128,
    }
    assert sampler["terms"]["gt_negative"]["pool_batch_sizes"] == {
        "offline_gt_negative": 128,
    }


def test_discriminator_config_resolves_standalone() -> None:
    cfg = _compose_config("discriminator")

    assert cfg.seed == 0
    assert cfg.env.environment == "PickPlaceCereal"
    assert cfg.algorithm.discriminator.task_name == "PickPlaceCereal"
    assert cfg.algorithm.discriminator.checkpoint is None
    assert cfg.algorithm.discriminator.learner_device == "cuda:0"


def test_train_dipole_preserves_online_config_contract() -> None:
    cfg = _compose_config("train_dipole")

    assert cfg.seed == 42
    assert cfg.env.environment == "PickPlaceBread"
    assert cfg.algorithm.discriminator.task_name == "PickPlaceBread"
    assert cfg.algorithm.discriminator.checkpoint is None
    assert cfg.algorithm.discriminator.learner_device == "cuda:0"
    assert cfg.logging.output_root == "./outputs/DIPOLE_rl"
    assert cfg.logging.tensorboard_dir == "tensorboard"
    assert cfg.logging.use_tensorboard is True
    assert cfg.logging.use_wandb is False


def test_train_dipole_rl_keeps_discriminator_on_vast_device() -> None:
    cfg = _compose_config("train_dipole_rl")

    assert cfg.algorithm.vast.config.device == "cuda:1"
    assert cfg.algorithm.discriminator.learner_device == "cuda:1"


def test_train_offline_dipole_preserves_discriminator_overrides() -> None:
    cfg = _compose_config("train_offline_dipole")

    assert cfg.env.environment == "PickPlaceCereal"
    assert cfg.algorithm.discriminator.task_name == "PickPlaceCereal"
    assert cfg.algorithm.discriminator.checkpoint is None
    assert cfg.algorithm.discriminator.learner_device == "cuda:0"
    assert cfg.algorithm.discriminator.inference.device == "cuda:0"
    assert cfg.algorithm.discriminator.inference.intervene_env is False
    assert cfg.algorithm.discriminator.hud.enabled is False
