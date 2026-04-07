from __future__ import annotations

import hydra
from omegaconf import DictConfig

from robosuite.discriminator.lpb_score.app.visualize import run_visualize


@hydra.main(version_base="1.2", config_path="./config", config_name="visualize")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for DSM failure visualization."""
    run_visualize(cfg)


if __name__ == "__main__":
    main()
