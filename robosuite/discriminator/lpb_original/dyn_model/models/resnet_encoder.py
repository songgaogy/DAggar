import torch
import hydra
import dill
from einops import rearrange
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torchvision
from typing import Optional, Sequence
from pathlib import Path

class ResNetEncoder(torch.nn.Module):
    def __init__(self, policy_ckpt_path: Optional[str], view_names: Sequence[str]):
        super().__init__()
        self.policy_ckpt_path = policy_ckpt_path
        self.view_names = list(view_names)
        self.emb_dim = 512
        self.latent_ndim = 2
        self.name = 'resnet'

        self.obs_encoder = torch.nn.ModuleDict()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if self.policy_ckpt_path:
            with open(self.policy_ckpt_path, 'rb') as f:
                payload = torch.load(f, pickle_module=dill)
                cfg = payload['cfg']
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(payload['cfg'], output_dir='debug_obs_encoder')
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            policy.to(device)
            policy.eval()

            for view_name in self.view_names:
                self.obs_encoder[view_name] = policy.obs_encoder.obs_nets[view_name].backbone

            del workspace
            del policy
            torch.cuda.empty_cache()
        else:
            for view_name in self.view_names:
                # keep per-view modules distinct to avoid surprising weight sharing
                self.obs_encoder[view_name] = self._build_torchvision_resnet18_backbone(pretrained=True)

    @staticmethod
    def _build_torchvision_resnet18_backbone(pretrained: bool) -> torch.nn.Module:
        # Prefer loading local weights to avoid network downloads.
        local_weights_path = None
        if pretrained:
            try:
                here = Path(__file__).resolve()
                for p in [here.parent] + list(here.parents):
                    candidate = p / "data" / "pretrained" / "resnet18-f37072fd.pth"
                    if candidate.is_file():
                        local_weights_path = candidate
                        break
            except Exception:
                local_weights_path = None

        if local_weights_path is not None:
            model = torchvision.models.resnet18(weights=None)
            state_dict = torch.load(str(local_weights_path), map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                model = torchvision.models.resnet18(weights=weights)
            except Exception:
                # Fall back to random init if weights cannot be fetched (e.g., offline / restricted network).
                try:
                    model = torchvision.models.resnet18(weights=None)
                except Exception:
                    model = torchvision.models.resnet18(pretrained=False)
        return torch.nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        view_embs = {}
        for view_name in self.view_names:
            imgs = x[view_name]
            b = imgs.shape[0]
            imgs = rearrange(imgs, "b t ... -> (b t) ...")
            imgs_emb = self.obs_encoder[view_name](imgs)
            imgs_emb = self.avgpool(imgs_emb)
            imgs_emb = imgs_emb.squeeze(-1).squeeze(-1)
            imgs_emb = imgs_emb.unsqueeze(1) # dummy patch dim
            imgs_emb = rearrange(imgs_emb, "(b t) p d -> b t p d", b=b)
            view_embs[view_name] = imgs_emb
        return view_embs


if __name__ == "__main__":
    ckpt = 'data/outputs/2025.03.26/04.49.31_train_diffusion_unet_hybrid_tool_hang_image_abs/checkpoints/240.ckpt'
    encoder = ResNetEncoder(ckpt, ['sideview', 'robot0_eye_in_hand'])
    bs = 2
    x_view_1 = torch.randn(bs, 2, 3, 128, 128).to('cuda')
    x_view_2 = torch.randn(bs, 2, 3, 128, 128).to('cuda')
    x = {'sideview': x_view_1, 'robot0_eye_in_hand': x_view_2}
    view_embs = encoder(x)
    for view_name, emb in view_embs.items():
        print(f"{view_name}: {emb.shape}")

