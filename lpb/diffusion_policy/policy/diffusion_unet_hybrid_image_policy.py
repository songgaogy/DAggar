import os
from typing import Dict

import yaml
from diffusion_policy.common.language_models import extract_text_features, get_text_model
import hydra
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

def boundary_penalty(action, lower_bound=-1.0, upper_bound=1.0):
    penalty = torch.relu(action - upper_bound) + torch.relu(lower_bound - action)
    return penalty.sum()


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(
            self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            if key == 'language':
                continue
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
        
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
            if 'language' in shape_meta['obs']:
                global_cond_dim += 32

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.dynamics_model_normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.correct_num = 0

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        ## =========================== load language model ===========================
        if 'language' in shape_meta['obs']:
            self.text_model, self.tokenizer, self.max_length = get_text_model(
                'libero_10', 'clip'
            )
        self._pending_guidance_target_info = None
        self._last_guidance_metrics = None

    def initialize_planner(self,
                           planner_target,
                           demo_dataset_config,
                           dynamics_model_ckpt,
                           action_step,
                           output_dir,
                           guidance_start_timestep,
                           guidance_scale,
                           threshold,
                           demo_dataset_path=None,
                           demo_batch_size=64,
                           demo_loader_workers=0,
                           demo_subsample_stride=1,
                           demo_max_samples=None,
                           nn_chunk_size=2048):
        planner_cls = hydra.utils.get_class(planner_target)

        # (gaoyuan) initialize the planner; see @./lpb/dyn_model/planner.py for detail
        self.planner = planner_cls(
            demo_dataset_config,
            dynamics_model_ckpt,    # planner takes pretraind dynamic model
            action_step,
            output_dir,
            demo_dataset_path,
            demo_batch_size,
            demo_loader_workers,
            demo_subsample_stride,
            demo_max_samples,
            nn_chunk_size,
        )
        self.guidance_start_timestep = guidance_start_timestep
        self.guidance_scale = guidance_scale
        self.planner.set_policy_action_normalizer(self.normalizer['action'])
        self.threshold = threshold

    def reset(self):
        self._pending_guidance_target_info = None
        self._last_guidance_metrics = None

    def _predict_next_visual_latent_from_sample(
            self,
            sample: torch.Tensor,
            current_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        action_sample = sample[..., :self.action_dim]
        expected_steps = self.planner.horizon * self.planner.frameskip
        init_actions_normalized = action_sample[:, 1:1 + expected_steps]
        if init_actions_normalized.shape[1] != expected_steps:
            raise RuntimeError(
                f"Expected {expected_steps} normalized action steps for dynamics rollout, "
                f"got {init_actions_normalized.shape[1]}"
            )

        with torch.no_grad():
            init_actions_unnormalized = self.normalizer['action'].unnormalize(init_actions_normalized)
            init_actions = self.planner.dyn_model_normalizer['act'].normalize(init_actions_unnormalized)
            action_batch = rearrange(
                init_actions,
                'b (h f) a -> b h (f a)',
                f=self.planner.frameskip,
                h=self.planner.horizon,
            )
            batch_size = action_batch.shape[0]
            current_obs_wm = self.planner.prepare_obs(current_obs, batch_size)
            act_0 = action_batch[:, :1, :]
            z = self.planner.dyn_model.encode(current_obs_wm, act_0)
            z_pred = self.planner.dyn_model.predict(z)
            z_new = z_pred[:, -1:, ...]
            z_obs, _ = self.planner.dyn_model.separate_emb(z_new)
            next_visual_latent = self.planner._flatten_visual_latent(z_obs['visual'].squeeze(1))
        return next_visual_latent

    def _compute_steering_comparison_metrics(
            self,
            guided_sample: torch.Tensor,
            base_sample: torch.Tensor,
            current_obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        guided_next_visual_latent = self._predict_next_visual_latent_from_sample(guided_sample, current_obs)
        base_next_visual_latent = self._predict_next_visual_latent_from_sample(base_sample, current_obs)

        with torch.no_grad():
            guided_reward, _ = self.planner.compute_nn_reward(guided_next_visual_latent)
            base_reward, _ = self.planner.compute_nn_reward(base_next_visual_latent)

            start = self.n_obs_steps - 1
            end = start + self.n_action_steps
            guided_action = self.normalizer['action'].unnormalize(guided_sample[..., :self.action_dim])[:, start:end]
            base_action = self.normalizer['action'].unnormalize(base_sample[..., :self.action_dim])[:, start:end]
            action_delta = guided_action - base_action

            metrics = {
                "guided_predicted_next_demo_distance": (-guided_reward).detach().cpu(),
                "base_predicted_next_demo_distance": (-base_reward).detach().cpu(),
                "predicted_next_demo_distance_improvement": ((-base_reward) - (-guided_reward)).detach().cpu(),
                "guided_base_predicted_next_latent_distance": torch.norm(
                    guided_next_visual_latent - base_next_visual_latent, dim=-1
                ).detach().cpu(),
                "guided_base_action_distance": reduce(
                    action_delta ** 2, 'b t a -> b', 'sum'
                ).sqrt().detach().cpu(),
                "guided_base_action_max_abs_distance": action_delta.abs().amax(
                    dim=tuple(range(1, action_delta.ndim))
                ).detach().cpu(),
            }
        return metrics
        
    # ========= inference  ============
    def guided_conditional_sample(
            self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            classifier_guidance=False,
            current_obs=None,
            text_latents=None,
            initial_trajectory=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        Run reverse diffusion to produce an action trajectory, optionally with LPB steering.

        The reverse process always follows the same backbone:

        1. Start from a noisy action trajectory.
        2. Re-apply observation conditioning at every diffusion step.
        3. Predict the denoising residual with the diffusion model.
        4. Step the diffusion scheduler toward a cleaner trajectory.

        When `classifier_guidance=True`, an extra LPB update is inserted before
        the scheduler step whenever the current observation is farther than the
        configured threshold from the demo latent manifold:

        1. Convert the current noisy sample into `pred_original_sample`, which is
           the current clean-action estimate for this denoising step. (sample is still noise
           but we directly compute clean output of this)
        2. Feed the first action chunk of that estimate into the dynamics model
           together with the current real observation.
        3. Predict the one-step-ahead latent and compute its nearest-demo
           distance in latent space.
        4. Backpropagate that scalar cost to the noisy trajectory and shift the
           denoising direction (current step!) before continuing reverse diffusion.

        Notes:
        - The steering target used in this loss is recomputed from the predicted
          next latent at each guided diffusion step. It is not the same as the
          fixed target used later for post-hoc evaluation metrics.
        - `initial_trajectory` allows guided and unguided sampling to start from
          identical noise, which makes their final action difference a direct
          measure of steering strength rather than diffusion randomness.
        """
        if text_latents is not None:
            current_obs['language'] = text_latents

        model = self.model
        scheduler = self.noise_scheduler

        if initial_trajectory is None:
            trajectory = torch.randn(
                size=condition_data.shape, 
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator)
        else:
            trajectory = initial_trajectory.to(
                device=condition_data.device,
                dtype=condition_data.dtype,
            ).clone()
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        if classifier_guidance:
            current_cost = -1 * self.planner.compute_current_reward(current_obs)    # cost = min_distance
            current_cost = current_cost.item()
            if current_cost >= self.threshold:
                self.correct_num += 1
        
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            trajectory = trajectory.detach().requires_grad_()

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)     # base diffusion policy

            # (gaoyuan) only too far did LPB start denoising loop
            if classifier_guidance and t < self.guidance_start_timestep and current_cost > self.threshold:
                trajectory0 = scheduler.step(model_output, t, trajectory).pred_original_sample
                loss = self.planner.compute_loss(trajectory0, current_obs)
                cond_grad = -torch.autograd.grad(loss, trajectory)[0]
                guidance_scale = self.guidance_scale
                grad_scale = guidance_scale * (1 - scheduler.alphas_cumprod[t]).sqrt()
                trajectory = trajectory.detach() + grad_scale * cond_grad

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], language_goal=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        text_latents = None
        if language_goal is not None:
            text_tokens = self.tokenizer(
                language_goal,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            text_latents = extract_text_features(
                self.text_model,
                text_tokens,
                language_emb_model='clip',
            )

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        with torch.no_grad():
            nsample = self.guided_conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                current_obs=dict_apply(obs_dict, lambda x: x[:, -1:, ...]),
                **self.kwargs)
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def predict_action_dyn_guided(self, obs_dict: Dict[str, torch.Tensor], language_goal=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        text_latents = None
        if language_goal is not None:
            text_tokens = self.tokenizer(
                language_goal,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            text_latents = extract_text_features(
                self.text_model,
                text_tokens,
                language_emb_model='clip',
            )

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        current_obs = dict_apply(obs_dict, lambda x: x[:, -1:, ...])
        guidance_target_info = self.planner.get_guidance_target_info(current_obs)
        initial_trajectory = torch.randn(
            size=cond_data.shape,
            dtype=dtype,
            device=device,
        )
        base_nsample = self.guided_conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            classifier_guidance=False,
            current_obs=current_obs,
            text_latents=text_latents,
            initial_trajectory=initial_trajectory,
            **self.kwargs)
        self._pending_guidance_target_info = {
            "target_demo_idx": guidance_target_info["target_demo_idx"].detach().clone(),
            "current_target_distance": guidance_target_info["current_target_distance"].detach().clone(),
            "num_inference_steps": int(self.num_inference_steps),
        }

        nsample = self.guided_conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            classifier_guidance=True,
            current_obs=current_obs,
            text_latents=text_latents,
            initial_trajectory=initial_trajectory,
            **self.kwargs)
        comparison_metrics = self._compute_steering_comparison_metrics(
            guided_sample=nsample,
            base_sample=base_nsample,
            current_obs=current_obs,
        )
        self._pending_guidance_target_info["steering_comparison_metrics"] = comparison_metrics
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def compute_actual_next_state_guidance_metrics(self, obs_dict: Dict[str, torch.Tensor]):
        if self._pending_guidance_target_info is None:
            return None

        next_obs = dict_apply(obs_dict, lambda x: x[:, -1:, ...])
        target_demo_idx = self._pending_guidance_target_info["target_demo_idx"]
        actual_next_distance = self.planner.compute_distance_to_demo_target(
            next_obs,
            target_demo_idx,
        )
        metrics = {
            "actual_next_target_distance": actual_next_distance.detach().cpu(),
            "current_target_distance": self._pending_guidance_target_info["current_target_distance"].detach().cpu(),
            "target_demo_idx": target_demo_idx.detach().cpu(),
            "num_inference_steps": self._pending_guidance_target_info["num_inference_steps"],
        }
        if "steering_comparison_metrics" in self._pending_guidance_target_info:
            metrics.update(self._pending_guidance_target_info["steering_comparison_metrics"])
        self._last_guidance_metrics = metrics
        self._pending_guidance_target_info = None
        return metrics
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        text_latents = None
        if 'language' in batch['obs']:
            if "language" in batch["obs"]:
                language_goal = batch["obs"]["language"]
                del batch["obs"]["language"]
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                text_latents = extract_text_features(
                    self.text_model,
                    text_tokens,
                    language_emb_model='clip',
                )
            elif "language_latents" in batch:
                text_latents = batch["language_latents"]

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
            if text_latents is not None:
                global_cond = torch.cat([global_cond, text_latents], dim=-1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
