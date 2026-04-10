# LPB Dice 模块总结

本文档总结 `robosuite/discriminator/lpb_dice/` 当前实现的目标、结构与运行方式。

## 1. 问题定义

`lpb_dice` 将原来的多分量负样本异常检测，改为：

- 使用冻结的策略编码器得到 `z_t`
- 使用已训练的 latent dynamics/world model 得到 transition feature
- 将 `z_t` 与 dynamics feature 拼接，作为 PU detector 的输入
- 使用 `expert + success_rollout` 作为正样本
- 使用 `fail_rollout` 作为 contaminated background
- 在校准阶段做 PU correction，并输出最终逐步 failure score

最终逐步分数包含：

- `pu_raw_score`
- `pu_corrected_score`
- `support_penalty`
- `final_step_score`

其中支持惩罚项由 `support_penalty.weight` 控制，默认 `0.0`。

## 2. 目录结构

- `core/`
  - `dataset.py`：多任务数据切分、latent cache、trajectory loading
  - `model.py`：latent dynamics backbone
  - `trainer.py`：world model trainer
  - `representation.py`：冻结 backbone 的 transition feature 提取
  - `pu_detector.py`：PU head 训练、校准、checkpoint load/save、trajectory scoring
  - `support.py`：support penalty 标定
- `app/`
  - `pipeline.py`：encoder / split / dataset 装配
  - `suboptimal.py`：suboptimal 可视化辅助编码
  - `train.py`：单一训练入口，负责 backbone reuse/train + PU detector train + calibration
  - `visualize.py`：单一可视化入口
- 顶层入口
  - `train.py`
  - `visualize_failures.py`
- `config/`
  - `train.yaml`
  - `visualize.yaml`
- `scripts/`
  - `train_lpb_dice.sh`
  - `visualize_lpb_dice_failures.sh`

## 3. 训练流程

1. 构建 latent cache
2. 若 `model.backbone_ckpt` 提供，则直接复用 dynamics checkpoint
3. 否则先训练 latent dynamics backbone
4. 冻结 backbone，提取每个 transition 的
   - `current_latent z_t`
   - `dynamics feature`
5. 拼接为 PU head 输入
6. 训练 positive-vs-background classifier
7. 在 calibration positive/background 上估计
   - `c_estimate`
   - background positive rate
   - support penalty 归一化统计
   - lambda threshold
8. 保存一个 combined `lpb_dice` checkpoint 供 visualize 直接加载

## 4. 可视化流程

可视化不再依赖独立 analyse/eval。

- 直接加载 `lpb_dice` combined checkpoint
- 对 fail rollout 或 suboptimal 轨迹逐步打分
- 输出 MP4 / PDF / `summary.json`
- PDF 中展示：
  - lambda vs threshold
  - raw / corrected / support curves

## 5. 当前设计特点

- 保留了原先好用的 latent cache、多任务支持与轨迹级可视化
- 删除了旧的 KNN 分支、policy chunk 分支与独立 analyse/eval 入口
- detector 核心现在是：
  - frozen representation
  - PU classifier
  - calibration-based correction
  - optional support penalty

## 6. 当前默认假设

- `support_penalty.weight = 0.0`
- `fail_rollout` 不是 clean negative，而是 contaminated background
- `current_latent z_t + dynamics feature` 是唯一 detector 输入
- `train` 与 `visualize` 是仅保留的用户工作流

