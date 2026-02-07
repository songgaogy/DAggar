import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pathlib
import datetime

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("now", lambda pattern: datetime.datetime.now().strftime(pattern), replace=True)


@hydra.main(config_path='./config', version_base=None, config_name="rollout_armada_robosuite")
def main(cfg: DictConfig):
    print(f"Starting Robosuite Rollout with config: {cfg.train.task.name}")
    if not os.path.isabs(cfg.checkpoint_path):
        cfg.checkpoint_path = os.path.join(ROOT_DIR, cfg.checkpoint_path)
    
    if 'device_ids' in cfg:
        if isinstance(cfg.device_ids, str):
            device_ids = [int(x) for x in cfg.device_ids.split(",")]
        elif isinstance(cfg.device_ids, int):
            device_ids = [cfg.device_ids]
        else:
            device_ids = list(cfg.device_ids)
    else:
        device_ids = [0]
    print(f"Using device ids: {device_ids}")
    
    try:
        from env_runner.robosuite_runner import RobosuiteRunner
        runner = RobosuiteRunner(cfg, rank=0, device_ids=device_ids)
        
        # start rollout
        runner.run_rollout()
        
    except Exception as e:
        print(f"Error during rollout: {e}")
        raise e


if __name__ == "__main__":
    main()