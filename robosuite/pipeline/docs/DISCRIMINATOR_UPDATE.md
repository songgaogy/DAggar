## Status
## TODOs
Every time when a TODO term is finished, please add a brief introduction under that term.

note: make sure the code is modular and standard, do NOT make one script extremely big.

- [x]: step-1. implement code framework and baseline method.
    1. Overall structure: from original pretrained nnPU discriminator checkpoint, mix online data and discriminator pertrain data to finetune the discriminator.
    2. extract data: since pretraind data is not extracted, so please implement `robosuite/discriminator/dyn_disc/scripts/extract_pretrain_data.sh` and `robosuite/discriminator/dyn_disc/utils/extract_data.py`. note that discriminator checkpoint is trained from `robosuite/discriminator/dyn_disc/scripts/run_pu_bce_robosuite_benchmark.sh`. Save data to `./data/<task_name>/discriminator-pretrain`, in the format that can be easily load and used by downstream scripts.
    3. finetune: modify the codebase `robosuite/pipeline/offline`, add discriminator finetuning logic. The offline data should be used as: only policy rollout data should be used for discriminator update; similar logic for VAST finetuning, offline data is arranged/separeted by human intervention and episode ends; only segments that end with "success"(successfully finish the episode) go to $D^+$. **This data usage design is important, please chat with me and let me confirm you understand it correctly**.
    4. note: freeze dynamic encoder, use the same transformer layer-1 latent; do NOT integrate discriminator finetuning logic into `robosuite/pipeline/offline/scripts/train_offline_dipole.sh` first, separate it to make further modification easier, and bash entrance should be `robosuite/pipeline/offline/scripts/finetune_disc.sh` and `robosuite/pipeline/offline/scripts/vis_disc_finetuned.sh`. make sure the code is moduler and extandable.
    
    Implementation: the checkpoint-aware extractor writes atomic latent shards and a reproducibility manifest for disjoint `positive_train`, `positive_calib`, and `unlabeled_train` pools, including dynamics-model and normalizer SHA-256 contracts. The standalone offline path separates policy/human sections, admits only policy-ended success sections to D+, excludes human and counterfactual actions, and sends all other policy sections to U with boundary-local action-chunk padding. Finetuning freezes the encoder, warm-starts the existing nnPU head, combines pretrain and online P/U pools, inherits the checkpoint nnPU/calibration semantics, and writes a backward-compatible `pu_bce_head_finetuned.pth` with provenance and TensorBoard history. Independent Hydra, extraction, finetuning, and finetuned-only visualization launchers are provided; no formal extraction, training, benchmark evaluation, or visualization was run during framework implementation.

- [x]: step-2. isolate offline discriminator finetuning under `pipeline`.
    The uncommitted warm-start implementation is split into pipeline-owned episode routing, latent encoding and pool loading, frozen-encoder contract, nnPU optimization, checkpoint, and visualization modules. The committed discriminator detector and benchmark visualizer remain unchanged except for the separately approved extraction provenance and `USE_CHUNK` contract. CUDA-only execution, P/U routing, optimizer and calibration semantics, Bash entrances, Hydra keys, and the finetuned checkpoint schema are preserved.

- [ ]: step-3. Finetuning discriminator with "gt-fail" data. (No code modification required, brainstorm)
    1. Obversation: Currently, offline data can provide very few "gt-fail" data; if not use gt fail data, and use current `robosuite/pipeline/offline/scripts/finetune_disc.sh` PU learning method, discriminator finetuning performance is very bad (no improvement at all)
    2. Why PU learning finetuning is useless? Because offline data are something that "hard to discriminate"(I ran discriminator to collect offline data, the offline data is something that pretrained discriminator cannot discriminate). 
    3. Please think of possible method that can work.
