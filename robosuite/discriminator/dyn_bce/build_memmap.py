from __future__ import annotations

import hydra
from omegaconf import DictConfig

from robosuite.discriminator.dyn_bce.modules.flow_encoder import FrozenFlowMultitaskEncoder
from robosuite.discriminator.dyn_bce.utils.dataset import build_datasets


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg: DictConfig) -> None:
    encoder = FrozenFlowMultitaskEncoder(
        checkpoint_path=str(cfg.policy.ckpt),
        device=str(cfg.policy.device),
        image_size=int(cfg.data.image_size),
        batch_size=int(cfg.policy.encoder_batch_size),
    )
    try:
        _, task_to_index, metadata = build_datasets(
            cfg_data=cfg.data,
            cfg_labels=cfg.labels,
            encoder=encoder,
            seed=int(cfg.seed),
        )
        print(f"dyn_bce memmap task_to_index={task_to_index}")
        print(f"dyn_bce memmap metadata={metadata}")
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
