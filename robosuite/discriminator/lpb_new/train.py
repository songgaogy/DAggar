from __future__ import annotations

import hydra
from omegaconf import DictConfig

from robosuite.discriminator.lpb_new.app.train import run_train


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for LPB world-model training."""
    run_train(cfg)


if __name__ == "__main__":
    main()
