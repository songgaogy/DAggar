"""Hydra CLI entry for LPB score DSM training.

Resolves ``robosuite/discriminator/lpb_score/config/train.yaml`` and forwards to
``run_train`` in ``app/train.py`` (data wiring, model build, ``Trainer.fit``).
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from robosuite.discriminator.lpb_score.app.train import run_train


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg: DictConfig) -> None:
    """Load Hydra config and run the full training pipeline."""
    run_train(cfg)


if __name__ == "__main__":
    main()
