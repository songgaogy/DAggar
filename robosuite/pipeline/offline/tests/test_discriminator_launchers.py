"""Smoke tests for standalone discriminator launcher defaults."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "robosuite" / "pipeline" / "offline" / "scripts"


def test_discriminator_launchers_parse_with_safe_environment_defaults() -> None:
    finetune = SCRIPTS / "finetune_disc.sh"
    visualize = SCRIPTS / "vis_disc_finetuned.sh"

    for script in (finetune, visualize):
        subprocess.run(["bash", "-n", str(script)], check=True)

    finetune_text = finetune.read_text(encoding="utf-8")
    assert 'NNPU_ENCODER_CKPT="${NNPU_ENCODER_CKPT:-' in finetune_text
    assert 'NNPU_CAMERA_TO_VIEW="${NNPU_CAMERA_TO_VIEW:-}"' in finetune_text
    assert 'USE_ONLY_OFFLINE="${USE_ONLY_OFFLINE:-true}"' in finetune_text
    assert '"offline.discriminator_finetune.use_only_offline=${USE_ONLY_OFFLINE}"' in finetune_text

    visualize_text = visualize.read_text(encoding="utf-8")
    assert 'MODEL_CKPT="${MODEL_CKPT:-' in visualize_text
    assert 'NUM_TRAJS="${NUM_TRAJS:-10}"' in visualize_text
