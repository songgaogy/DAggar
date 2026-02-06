import sys
# Use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf, open_dict
import pathlib
import os
import torch
import copy
import datetime
from torch.multiprocessing import Process

ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
except ImportError as e:
    print("Could not import BaseWorkspace. Please check your python path.")
    raise e

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("now", lambda pattern: datetime.datetime.now().strftime(pattern), replace=True)


def main(rank, cfg: OmegaConf, device_ids):
    """
    Main training function compatible with Multi-GPU (DDP).
    """
    world_size = len(device_ids)
    device_id = device_ids[rank]
    device = f"cuda:{device_id}"
    print(f"[Rank {rank}] Initializing process on {device}...")

    # Initialize Distributed Process Group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # instantiate workspace
    try:
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg, rank, world_size, device_id, device)
    except Exception as e:
        print(f"[Rank {rank}] Error initializing workspace class {cfg._target_}: {e}")
        raise e

    # run
    workspace.run(rank, world_size, device_id, device)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_name>")
        print("Example: python train.py train_armada_robosuite")
        sys.exit(1)

    config_name = sys.argv[1]
    print(f"Loading configuration: {config_name}")

    with hydra.initialize(version_base=None, config_path='./config'):
        cfg = hydra.compose(config_name=config_name)

    # output
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(ROOT_DIR, "outputs", cfg.task_name, run_id))
    os.makedirs(run_dir, exist_ok=True)
    with open_dict(cfg):
        cfg.output_dir = run_dir
        print(f"results will be saved to: {run_dir}")

    # for multi-GPU
    if 'device_ids' in cfg:
        if isinstance(cfg.device_ids, str):
            device_ids = [int(x) for x in cfg.device_ids.split(",")]
        elif isinstance(cfg.device_ids, int):
            device_ids = [cfg.device_ids]
        else:
            device_ids = list(cfg.device_ids)
    else:
        device_ids = [0]
    
    print(f"Training on devices: {device_ids}")

    # Setup Distributed Environment Variables
    os.environ["MASTER_ADDR"] = "localhost"
    
    # Find a free port for DDP communication
    port = 30003
    import socket
    found_port = False

    for i in range(100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                found_port = True
                break
        except OSError:
            port += 1
    
    if not found_port:
        raise RuntimeError("Could not find a free port for DDP communication.")
        
    print(f"Using Master Port: {port}")
    os.environ["MASTER_PORT"] = f"{port}"

    # Launch Processes
    if len(device_ids) == 1:
        main(0, cfg, device_ids)
    elif len(device_ids) > 1:
        OmegaConf.resolve(cfg)
        processes = []
        for rank in range(len(device_ids)):
            p = Process(target=main, args=(rank, cfg, device_ids))
            p.start()     
            processes.append(p)
        
        for p in processes:
            p.join()
            