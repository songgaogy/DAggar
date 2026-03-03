import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import datetime
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from utils.env_util import PandaLiftProprioExtractor
from datasets import PandaLiftFlowDataset
from flow import FlowPolicy


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment_images(images, pad=4):
    """
    Random shift / crop with strict temporal consistency.
    A single crop offset is sampled per batch element and shared across history frames.
    Expects images of shape: [B, K, C, H, W]
    """
    B, K, C, H, W = images.shape

    images_flat = images.view(B * K, C, H, W)
    images_padded_flat = F.pad(images_flat, (pad, pad, pad, pad), mode="replicate")
    images_padded = images_padded_flat.view(B, K, C, H + 2 * pad, W + 2 * pad)

    w_start = torch.randint(0, 2 * pad + 1, (B,), device=images.device)
    h_start = torch.randint(0, 2 * pad + 1, (B,), device=images.device)

    cropped_images = torch.empty((B, K, C, H, W), device=images.device)

    for i in range(B):
        hs = int(h_start[i].item())
        ws = int(w_start[i].item())
        cropped_images[i] = images_padded[i, :, :, hs : hs + H, ws : ws + W]

    return cropped_images


@hydra.main(version_base="1.2", config_path="./config", config_name="train_flow")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = to_absolute_path(cfg.data_dir)
    save_dir = to_absolute_path(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    extractor = PandaLiftProprioExtractor(
        robots="Panda",
        env_name="Lift",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        camera_names=None,
        reward_shaping=False,
    )

    ds = PandaLiftFlowDataset(
        data_dir=data_dir,
        proprio_extractor=extractor,
        camera_name=cfg.camera,
        history_len=cfg.history_len,
        horizon=cfg.chunk_size,
        stride=cfg.stride,
        image_size=cfg.image_size,
        normalize=True,
        cache_proprio=True,
    )

    dl_kwargs = dict(
        dataset=ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        persistent_workers=bool(cfg.train.num_workers > 0),
        pin_memory=True,
        drop_last=True,
    )
    if cfg.train.num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(getattr(cfg.train, "prefetch_factor", 4))
    dl = DataLoader(**dl_kwargs)

    act_dim = ds[0]["actions"].numel()
    prop_dim = ds[0]["proprio"].numel()

    model = FlowPolicy(
        act_dim=act_dim,
        proprio_in_dim=prop_dim,
        img_dim=cfg.flow.image_dim,
        prop_dim=cfg.flow.propior_dim,
        time_dim=cfg.flow.time_dim,
        token_dim=cfg.flow.token_dim,
        pretrained_resnet=True,
        freeze_resnet=True,
        temporal_layers=cfg.flow.temporal_layers,
        temporal_heads=cfg.flow.temporal_heads,
        vel_hidden=cfg.flow.vel_hidden,
        vel_layers=cfg.flow.vel_layers,
        dropout=cfg.train.dropout,
        history_len=cfg.history_len,
        action_chunk_size=cfg.chunk_size,
        action_temporal_layers=cfg.flow.action_temporal_layers,
        action_temporal_heads=cfg.flow.action_temporal_heads,
    ).to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"), device=device)

    total_steps = cfg.train.epochs * len(dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    img_normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    global_step = 0
    unfreeze_ratio = float(getattr(cfg.train, "unfreeze_epoch_ratio", 0.2))
    unfreeze_epoch = int(unfreeze_ratio * cfg.train.epochs)
    should_unfreeze = bool(getattr(cfg.train, "unfreeze_backbone", False))
    did_unfreeze = False
    lambda_endpoint = float(getattr(cfg.train, "lambda_endpoint", 0.5))
    lambda_smooth = float(getattr(cfg.train, "lambda_smooth", 0.05))

    print("start training...")
    for ep in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        data_time_running = 0.0
        step_time_running = 0.0
        loop_end_time = time.time()

        if should_unfreeze and (not did_unfreeze) and (ep >= unfreeze_epoch):
            for p in model.img_enc.backbone.parameters():
                p.requires_grad = True
            did_unfreeze = True

        for it, batch in enumerate(dl):
            data_time = time.time() - loop_end_time
            iter_start = time.time()
            images = batch["images"].to(device, non_blocking=True)      # [B, K, C, H, W]
            
            B, K, C, H, W = images.shape
            images_aug = augment_images(images) 
            images_flat = images_aug.view(B * K, C, H, W)
            images_norm = img_normalize(images_flat).view(B, K, C, H, W)
            
            proprio = batch["proprio"].to(device, non_blocking=True)    # [B, prop_dim]
            x1 = batch["actions"].to(device, non_blocking=True)         # [B, act_dim]

            B_batch = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(B_batch, device=device)

            x_t = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
            v_target = x1 - x0

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=(device.type == "cuda"), device_type=device.type):
                v_pred = model(x_t=x_t, t=t, images=images_norm, proprio=proprio)
                flow_loss = torch.mean((v_pred - v_target) ** 2)
                x1_pred = x_t + (1.0 - t).unsqueeze(-1) * v_pred
                endpoint_loss = torch.mean((x1_pred - x1) ** 2)

                if cfg.chunk_size > 1:
                    step_dim = x1.shape[-1] // cfg.chunk_size
                    x1_seq = x1_pred.view(B_batch, cfg.chunk_size, step_dim)
                    smooth_loss = torch.mean((x1_seq[:, 1:] - x1_seq[:, :-1]) ** 2)
                else:
                    smooth_loss = torch.zeros((), device=device, dtype=x1.dtype)

                loss = flow_loss + lambda_endpoint * endpoint_loss + lambda_smooth * smooth_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            scheduler.step()
            ema_model.update_parameters(model)

            running += float(loss.item())
            global_step += 1
            iter_end = time.time()
            data_time_running += data_time
            step_time_running += (iter_end - iter_start)
            loop_end_time = iter_end

            if (it + 1) % cfg.log_every == 0:
                avg = running / cfg.log_every
                avg_data_ms = (data_time_running / cfg.log_every) * 1000.0
                avg_step_ms = (step_time_running / cfg.log_every) * 1000.0
                samples_per_sec = cfg.train.batch_size / max(1e-6, (step_time_running / cfg.log_every))
                running = 0.0
                data_time_running = 0.0
                step_time_running = 0.0
                print(
                    f"epoch={ep:04d} iter={it:05d} step={global_step:07d} "
                    f"loss={avg:.6f} flow={float(flow_loss.item()):.6f} "
                    f"endpoint={float(endpoint_loss.item()):.6f} smooth={float(smooth_loss.item()):.6f} "
                    f"data_ms={avg_data_ms:.2f} step_ms={avg_step_ms:.2f} samp_s={samples_per_sec:.1f}"
                )

        ckpt = dict(
            model=model.state_dict(),
            ema_model=ema_model.module.state_dict(),
            opt=opt.state_dict(),
            epoch=ep,
            cfg=vars(cfg),
            act_mean=ds.act_mean,
            act_std=ds.act_std,
            prop_mean=ds.prop_mean,
            prop_std=ds.prop_std,
        )

        if (ep + 1) % cfg.save_freq == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(save_dir, f"flow_policy_ep{(ep+1):04d}_{timestamp}.pt")
            torch.save(ckpt, path)
            print(f"saved: {path} elapsed={time.time()-t0:.1f}s")

    extractor.close()


if __name__ == "__main__":
    main()
