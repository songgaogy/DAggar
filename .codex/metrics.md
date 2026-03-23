## **Loss**
- `total_loss`
  总训练目标。等于
  `occ_loss + alpha_dyn * dyn_loss + eta_decor * decor_loss + xi_fuse * fuse_loss`
  在 [trainer.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/utils/trainer.py#L183)。

- `occ_loss`
  occupancy witness 的 nnPU loss。
  它不是普通 BCE，而是“正样本 + 未标注样本”的 PU learning 风险，在 [losses.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/utils/losses.py#L35)。

- `dyn_loss`
  dynamics ensemble 的 beta-NLL。
  因为里面有 `logvar` 项，所以它可以是负数，这不是 bug，在 [losses.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/utils/losses.py#L107)。

- `decor_loss`
  occupancy private branch 和 dynamics private branch 的 cross-covariance penalty。
  越小表示两个 private 表征越不相关，在 [losses.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/utils/losses.py#L132)。

- `fuse_loss`
  judge/fusion head 的加权 BCE loss。它学的是最终风险预测，不直接训练 witness 内部特征，在 [losses.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/utils/losses.py#L7)。

## **Occupancy branch**
- `occ_raw`
  batch 打印时显示的是 `occ_risk_mean`，也就是 raw occupancy risk 的均值。
  这个值来自
  `occ_prob = 1 - sigmoid(occ_logit)`
  在 [model.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/modules/model.py#L495)。

- `occ_risk_mean`
  和 `occ_raw` 是同一个量，只是 epoch 汇总名。

- `occ_positive_logit_mean`
  expert + success rollout 这些“正样本”在 occupancy head 上的平均 logit。
  越大表示模型越认为这些样本是正类。

- `occ_unlabeled_logit_mean`
  fail rollout 这个 unlabeled pool 的平均 logit。
  如果它和 `occ_positive_logit_mean` 太接近，说明 occupancy 分支区分不出 fail pool。

- `occ_positive_risk_mean`
  正样本上的 raw occupancy risk 均值。
  正常应偏低，因为正样本应该“不危险”。

- `occ_unlabeled_risk_mean`
  fail-unlabeled 样本上的 raw occupancy risk 均值。
  正常应高于 `occ_positive_risk_mean`。

- `occ_positive_term`
  nnPU 里的正样本风险项，来自 `positive_loss` 的加权均值，在 [losses.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/utils/losses.py#L63)。

- `occ_negative_from_positive`
  用正样本估计出来的“负风险校正项”。

- `occ_negative_from_unlabeled`
  从 fail-unlabeled pool 估计出来的负风险项。

- `occ_negative_risk_before_clamp`
  nnPU 里真正的 negative risk，在 clamp 前的值。
  如果它长期很负，说明模型在 train 上过度依赖 nnPU 的截断。

- `occ_negative_risk_after_clamp`
  clamp 后实际参与 loss 的 negative risk。
  在 nnPU 下会被截到 `>= 0`。

- `occ_nnpu_clamped`
  一个诊断量。接近 `1` 表示当前 batch 的 nnPU negative risk 经常被 clamp。
  接近 `0` 表示很少被 clamp。

## **Occupancy Metrics**
这里有两组，别混：

1. `occ_*`
   这是基于 raw `occ_prob` 算的分类指标。

2. `occ_evidence_*`
   这是基于 calibrator 后的 `occ_evidence` 算的分类指标。
   `occ_evidence = occ_calibrator(occ_prob).detach()` 在 [model.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/modules/model.py#L530)。

每组里这些字段含义相同：
- `accuracy`
  按阈值 `0.5` 二值化后的准确率。
- `precision`
  预测为正的样本里，真正正类的比例。
- `recall`
  真正正类里，被预测出来的比例。
- `f1`
  `precision` 和 `recall` 的调和平均。
- `positive_rate`
  模型预测为正的比例。
- `label_rate`
  数据真实正标签比例。
- `auroc`
  ROC AUC，衡量排序能力，和固定阈值无关。
- `auprc`
  PR AUC，在类别不平衡时很有用。
- `nonfinite_rate`
  这一批用于算指标的分数里，有多少是 NaN/Inf。
- `finite_rate`
  非 NaN/Inf 的比例。
这些都在 [metrics.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/utils/metrics.py#L15)。

## **Dynamics / EPI branch**
- `dyn_residual_mean`
  dynamics ensemble 平均预测和 EMA target 之间的残差均值。
  越大通常表示 transition 更异常，在 [model.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/modules/model.py#L523)。

- `dyn_evidence_mean`
  `dyn_residual` 经过 calibrator 后的均值。
  这是送给 judge 的 detached evidence。

- `epi_variance_mean`
  ensemble disagreement 的均值，也就是不同 dynamics head 之间分歧有多大，在 [model.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/modules/model.py#L525)。

- `epi_evidence_mean`
  `epi_variance` 经过 calibrator 后的均值。
  同样是 judge 的输入之一。

- `dyn_weight_mean`
  当前 batch 里 dynamics loss 的平均 sample weight。
  因为 fail rollout 前缀/后缀权重不同，所以它不一定等于 1。

## **Judge / Fusion index**
- `judge_prob`
  实际是 `sigmoid(judge_logit)`，在 [trainer.py](/home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/dyn_bce/utils/trainer.py#L217)。
- `judge_accuracy / precision / recall / f1 / auroc / auprc / positive_rate / label_rate`
  含义和上面二分类指标完全一致，只不过对象变成最终的 fusion 输出。

直观理解：
- `occ_*` 看 occupancy witness 自己强不强
- `judge_*` 看三个 witness 融合后最终强不强
- 如果 `judge` 比 `occ` 和 `dyn` 单独都更好，说明 fusion 是有价值的

## **data distribution**
- `risk_target_mean`
  当前 batch 里 soft risk label 的平均值。
  它反映整体“风险密度”有多高，不是硬标签比例。

- `label_rate`
  是 hard label 的正类比例。
  `risk_target_mean` 和 `label_rate` 不一定完全相同，因为 fail rollout 用的是软 schedule。
