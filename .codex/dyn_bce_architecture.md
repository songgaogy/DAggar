# dyn_bce Architecture

## 1. Goal

这个实现把 `dyn_bce` 做成一个标准的多任务离线判别器训练包，核心目标是：

- 复用 `flow_multi` 预训练策略编码器
- 在 `(s, a, s')` 级别做判别
- 同时利用 `expert / success_rollout / fail_rollout`
- 把 occupancy witness、dynamics witness、judge/fusion 清晰拆开

对应代码入口：

- `robosuite/discriminator/dyn_bce/train.py`
- `robosuite/discriminator/dyn_bce/config/train.yaml`
- `robosuite/run/train_dyn_bce.sh`

## 2. Pretrained encoder

预训练编码器封装在：

- `robosuite/discriminator/dyn_bce/flow_encoder.py`

这里直接加载你的多任务 checkpoint：

- `flow_multi_ep0100_20260320_114720.pt`

我最终选择把 `task_scene_cond` 当成主 latent `u_t`，原因是：

1. 它就是最终传给 policy head / U-Net 的条件表征。
2. 它已经融合了多相机视觉、proprio、语言。
3. 维度固定，适合做标准 witness network。
4. 比直接消费 `fusion` token 序列更稳定、工程复杂度更低。

所以实现里：

- `u_t = flow_multi.encode_context(images, proprio, language)`

编码器默认冻结，不做 end-to-end finetune。这样做有两个好处：

- 训练更稳
- 可以缓存整条轨迹的 latent，避免每个 epoch 反复跑 RGB encoder

## 3. Data pipeline

数据构建在：

- `robosuite/discriminator/dyn_bce/dataset.py`

### 3.1 显式的 trajectory budget

你要求显式写清每个 task、每个 data type 用多少 trajectory，我在 config 里按 task 给了：

- `train.num_expert_traj`
- `train.num_success_traj`
- `train.num_fail_traj`
- `val.*`
- `test.*`

当前默认是每个 task：

- train: `240 / 240 / 240`
- val: `30 / 30 / 30`
- test: `30 / 30 / 30`

也就是说每个 task、每个 data type 一共使用 300 条 trajectory，其中 240 条训练，60 条评估。

### 3.2 Split 粒度

split 是按 `(task, data_type)` 独立抽样的 trajectory-level split，不是全局混在一起随机切。

这样可以避免：

- task 之间分布互相污染
- 某个 data type 占据验证集
- 同一条 trajectory 的 transition 泄露到 train/test 两边

### 3.3 Transition form

每条轨迹先被编码成 latent 序列，然后构造：

- `(u_t, a_{t:t+H-1}, u_{t+H})`

当前默认：

- `transition_horizon = 1`

但配置里保留了 horizon 开关，后续可以直接扩成 chunk transition。

### 3.4 Fail weighting over the whole trajectory

fail rollout 不再只在 tail 50% 才加权。

现在对整条 fail trajectory 都定义一个 sigmoid schedule：

- `risk_target(t) = sigmoid((progress - center) / temperature)`
- judge/fusion weight 用整段 sigmoid 曲线提升
- occupancy PU 分支不再使用时间加权，直接看到原始 fail marginal

这里 `center` 仍然默认在 `0.5`，但它现在表示 sigmoid 的中心点，而不是“只从 50% tail 开始起作用”。

因此：

- 轨迹前段也有非零 failure risk target
- 越往后 judge weight 越大
- 整条 fail trajectory 都被纳入 judge 学习
- occupancy witness 则保持严格的 PU 设定，不让时间权重破坏 unlabeled 分布

### 3.5 Dynamics data usage

根据你的修改要求，dynamics model 现在只在：

- `expert`
- `success_rollout`

上训练。

对于 `fail_rollout`：

- `dyn_weight = 0`
- 完全不参与 dynamics loss

这意味着 fail data 只用于：

- occupancy witness
- fusion judge
- 最终 anomaly evidence 分布塑形

不会再被 nominal dynamics model 拿来拟合。

### 3.6 Latent cache

编码器输出会缓存到：

- `data/.dyn_bce_cache`

缓存 key 里包含：

- task name
- source hdf5 path
- demo key
- image size
- camera names
- source file mtime
- checkpoint path

这样第一次会花时间建 cache，后续复训就能直接复用。

## 4. Occupancy witness

主模型在：

- `robosuite/discriminator/dyn_bce/model.py`

Occupancy 分支结构是：

1. 共享 trunk 输入 `u_t + task_embedding`
2. 生成 shared latent `h_sh`
3. 单独生成 occupancy-private latent `h_occ_private`
4. occupancy head 从 `[h_sh, h_occ_private, task_embedding]` 输出一个 success-like occupancy logit

这里对应你的设计：

- `h_t^{sh} = B(P_sh(u_t))`
- `h_t^{occ,p} = P_occ(u_t)`
- `s_t^{occ} = D_occ(h_t^{sh}, h_t^{occ,p})`

### 4.1 关于你问的 “PU-learning like BCE discriminator”

你问得对。上一版代码里其实没有真正的 PU-learning occupancy discriminator。

上一版只是：

- 对 occupancy head 做了 weighted BCE
- expert/success 作为 0
- fail rollout 用 soft failure target

这并不是真正的 positive-unlabeled objective。

这次我已经把它改成了真正的 PU-like occupancy loss：

- positive set: `expert + success_rollout`
- unlabeled set: `fail_rollout`
- loss form: non-negative PU BCE (`nnPU`-style risk estimator)
- 关键约束：fail rollout 作为 unlabeled pool 时不做时间重加权

对应实现位置：

- `robosuite/discriminator/dyn_bce/losses.py` 中的 `occupancy_pu_loss`
- `robosuite/discriminator/dyn_bce/trainer.py` 中对 `occ_loss` 的调用

也就是说，现在 occupancy 头学的是：

- “这个状态是否属于 success-like occupancy”

然后在 forward 里把它转换成 failure risk：

- `occ_failure_prob = 1 - sigmoid(occ_logit)`

这样就同时满足了：

- occupancy 用正样本 vs 未标注样本的 PU 学习
- judge 仍然吃 failure-oriented evidence

## 5. Dynamics witness

Dynamics 部分用了 ensemble 结构，默认：

- `ensemble_size = 5`

### 5.1 Current branch

当前时刻输入 `u_t + task_embedding`，每个 ensemble member 都有自己的 private encoder：

- `dyn_private_encoder_m(u_t)`

得到：

- `h_dyn_private_m`

### 5.2 Action encoder

动作不是直接 flatten，而是用了一个小型 transformer action encoder：

- action projection
- learnable CLS token
- positional embedding
- transformer encoder

这样做是因为你说过数据量够，可以把网络做得更有表达力，而且 action 序列顺序本身有信息。

### 5.3 Predictor heads

每个 ensemble member 的 dynamics head 输入：

- `h_sh`
- `h_dyn_private_m`
- action feature
- task embedding

输出：

- predictive mean
- predictive log variance

所以 dynamics 不是简单点估计，而是 heteroscedastic predictive distribution。

## 6. EMA target branch

你特别提到要有 EMA target branch。现在这个 target branch 明确同时包含 shared state 和 dynamics-private state：

- online 模型里有 trainable `shared_encoder` 和 `dyn_private_encoders`
- 同时保留它们各自的 EMA copy
- `u_{t+H}` 走 EMA branch
- target 定义为 `[ema_shared(u_{t+H}), mean_m ema_dyn_private_m(u_{t+H})]`

也就是：

- current side: online 分支预测 shared + private 的联合下一状态
- target side: EMA 分支提供 shared + private 的联合目标

这样 dynamics loss 就会真正监督 shared trunk，而不是只监督 private dynamics space。

## 7. Three evidences

模型最终形成三路 evidence：

### 7.1 Occupancy evidence

occupancy head 内部学的是 success-like probability，但送给后续 judge 的是 failure risk：

- `occ_success_prob = sigmoid(occ_logit)`
- `occ_failure_prob = 1 - occ_success_prob`

这里不再对 occupancy probability 做第二次 running-stat + sigmoid 标定，而是直接把 detached 的 `occ_failure_prob` 作为 occupancy evidence。这样可以避免 double-sigmoid 扭曲。

### 7.2 Dynamics residual evidence

先把 ensemble predictive mean 取平均，再和 EMA target 做 residual：

- `dyn_residual = || mean(pred_m) - target_ema ||`

它对应 transition mismatch。

### 7.3 Epistemic evidence

对 ensemble predictive mean 做 member-wise variance：

- `epi_variance`

它对应 epistemic disagreement。

## 8. Calibration and judge

校准器也是在 `model.py` 里实现的。

当前的选择是：

- occupancy evidence 不再额外标定，直接用 detached `occ_failure_prob`
- dynamics residual 用 running mean / variance 做轻量标定
- epistemic variance 用 running mean / variance 做轻量标定
- 三路 evidence 全都 detach，再送给 judge

这样既保留了 judge 不反向操纵 witness 的结构，又避免 occupancy evidence 的数值语义被二次压缩。

Judge 只吃三维输入：

- raw occupancy failure evidence
- calibrated dynamics evidence
- calibrated epistemic evidence

Judge 本身是一个小 MLP，输出 `judge_logit`。

## 9. Losses

loss 实现在：

- `robosuite/discriminator/dyn_bce/losses.py`

总损失是：

- `L_occ`
- `alpha * L_dyn`
- `eta * L_decor`
- `xi * L_fuse`

### 9.1 Occupancy loss

现在 occupancy loss 已经是 PU-like objective，而不是普通 weighted BCE。

定义为：

- positives: `expert + success_rollout`
- unlabeled: `fail_rollout`

使用 non-negative PU 风格 risk：

- positive term: positives should be success-like
- unlabeled term: fail rollout is treated as unlabeled occupancy data
- unlabeled 项不使用时间重加权，保持 PU 风格的经验边缘分布
- `occupancy_positive_prior` 控制正类先验

当前默认：

- `occupancy_positive_prior = 0.67`
- `occupancy_nnpu = true`

这个 prior 取值和当前每个 task 的 data budget 一致：

- positive trajectories: 240 expert + 240 success = 480
- unlabeled trajectories: 240 fail
- 所以正类先验默认约为 `480 / (480 + 240) = 0.67`

### 9.2 Judge labels

judge 仍然直接对 failure risk target 做 weighted BCE。

也就是说：

- occupancy witness 用 PU 学 success occupancy
- judge 用 risk target 学最终 failure decision

### 9.3 Dynamics supervision

Dynamics loss 现在只由 `expert + success_rollout` 提供监督。

fail rollout 完全不进入 dynamics fitting。

### 9.4 Beta-NLL

Dynamics loss 用的是 beta-NLL，既考虑 prediction mean，也考虑 variance。

### 9.5 Decorrelation

`L_decor` 用 cross-covariance penalty，在 batch 里压低：

- occupancy-private latent
- mean dynamics-private latent

之间的线性相关性。

## 10. Trainer and evaluation

trainer 在：

- `robosuite/discriminator/dyn_bce/trainer.py`

它负责：

- dataloader
- AMP
- optimizer / grad clip
- EMA update
- val model selection
- test evaluation
- offline WandB logging
- checkpoint save
- summary json save

当前 best model 选择分数是：

- `judge_f1 + judge_auroc`

也就是说 model selection 更偏向最终 fuse/judge 效果，而不是单独某个 witness。

## 11. Files

修改涉及的主要文件有：

- `robosuite/discriminator/dyn_bce/losses.py`
- `robosuite/discriminator/dyn_bce/dataset.py`
- `robosuite/discriminator/dyn_bce/model.py`
- `robosuite/discriminator/dyn_bce/trainer.py`
- `robosuite/discriminator/dyn_bce/train.py`
- `robosuite/discriminator/dyn_bce/config/train.yaml`
- `.codex/dyn_bce_architecture.md`

## 12. Suggested next step

在你所有 rollout 数据都准备好以后，建议先做两件事：

1. 先把 config 里的 trajectory counts 和真实数据量核对一遍。
2. 先跑一个小 epoch smoke run，确认 cache 构建和显存占用都正常，再开始完整训练。
