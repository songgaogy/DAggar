if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil

from diffusion_policy.diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.diffusion_policy.model.common.lr_scheduler import get_scheduler

# Register eval resolver for OmegaConf to allow python expressions in yaml
OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainArmadaRobosuiteWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, rank, world_size, device_id, device='cuda:0,1', output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy).to(device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        if dist.is_available() and dist.is_initialized():
            self.model = DDP(self.model, device_ids=[device_id], find_unused_parameters=True)

        # ema
        self.ema_model: BaseImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # optimizer, adamW by default
        cfg.optimizer.learning_rate = cfg.optimizer.learning_rate * world_size 
        opt_cfg = dict(cfg.optimizer)
        opt_cfg.pop("_target_", None)
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = model_obj.get_optimizer(**opt_cfg)

        self.global_step = 0
        self.epoch = 0

    def run(self, rank, world_size, device_id, device):
        cfg = copy.deepcopy(self.cfg)

        # resume training logic
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path, exclude_keys=['optimizer'], include_keys=[])

        # pre-collected data
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)

        # distributed sampler for DDP
        sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.training.seed, drop_last=True)
        train_dataloader = DataLoader(dataset, sampler=sampler, **cfg.dataloader)
        
        # normalizer and init the model
        normalizer = dataset.get_normalizer()
        self.model.module.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.module.set_normalizer(normalizer)

        # Configure Validation Dataset (Optional)
        val_dataloader = None
        if cfg.task.dataset.get('val_ratio', 0) > 0:
            val_dataset = dataset.get_validation_dataset()
            sampler_val = DistributedSampler(dataset=val_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.training.seed, drop_last=True)
            val_dataloader = DataLoader(val_dataset, sampler=sampler_val, **cfg.val_dataloader)

        # lr
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps // world_size,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # env runner, float and STAC are included
        # TODO(gaoyuan)
        env_runner: BaseImageRunner
        if cfg.mode == "train":
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner_dummy,
                output_dir=self.output_dir
            )
        elif cfg.mode == "rollout":
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir
            )
        else:
            raise ValueError

        # wandb logging
        if rank == 0:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )

        # ckpt
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # setup training
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        train_sampling_batch = None

        # Debug mode settings
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            self.epoch = cfg.training.num_epochs

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", disable=(rank!=0), 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # Move batch to device
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # Apply augmentation if dataset supports it (common in Robomimic)
                        if hasattr(dataset, 'gpu_color_augment'):
                            batch = dataset.gpu_color_augment(batch)
                            
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # Compute Loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # Optimizer Step
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # Update EMA
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # Logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # Log frequently
                            if rank == 0:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # End of Epoch Aggregation
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # eval
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # 1. Run Rollout (Env Runner)
                if (self.epoch % cfg.training.rollout_every) == 0:
                    # The env_runner handles the robosuite loop internally
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                # 2. Run Validation Loss
                if val_dataloader is not None and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", disable=(rank!=0), 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

                # 3. Simple Action MSE Check
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        result = policy.module.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                
                # checkpointing
                if (self.epoch % cfg.training.checkpoint_every) == 0 and rank == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # Sanitize keys for filename
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                policy.train()

                # Final log for epoch
                if rank == 0:
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainArmadaRobosuiteWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()