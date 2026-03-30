# Dipole Pipeline Debug Notes

## 1. 这次我实际确认并修复了什么

这次我没有重写 `dipole` 目录，而是在尽量保持现有 pipeline 结构不变的前提下，做了两类高价值修正：

1. **修复训练恢复时的 buffer 兼容性问题**
   在 `train_dipole.py` 中，原先只检查了 `demo_buffer` 和当前 observation / action 结构是否兼容，但没有检查 `online_buffer`。
   如果你恢复的是一个旧 run，且那个 run 的 camera / observation 结构和当前任务不完全一致，那么程序可能在真正开始 rollout、往 online buffer 里追加新 transition 时才因为 shape mismatch 崩掉。
   现在我把 `online_buffer` 的兼容性检查也补上了，不兼容时会直接清空并打印 warning。

2. **优化 dipole 双头 flow 模型的重复条件编码**
   在 `robosuite/pipeline/algorithms/dipole/models/flow.py` 中，原实现每一个 ODE / flow step 都会重新做一次完整的多模态条件编码：
   - image encoder
   - proprio tokenizer
   - language encoder
   - language-guided modulation
   - fusion
   - aggregator

   但对于同一个 observation 来说，这些条件在一次 action chunk 采样过程中根本没有变化。
   所以我把这部分改成了：
   - **先编码一次 context**
   - 然后在所有 ODE steps 里复用这个 context

   这个优化同时作用于：
   - 在线推理时的 `select_action`
   - learner 更新时的 `update`

   对 dipole 尤其重要，因为它有 `pos` / `neg` 两个 head，原来在每一步都重复做整套条件编码，成本比较高。

3. **去掉 `_weighted_mean` 里的 CPU 同步**
   原来 `_weighted_mean` 里为了判断权重和是否接近 0，会把 tensor `.item()` 到 CPU。
   这会引入不必要的 host-device sync，尤其在 GPU 训练时会打断流水。
   现在改成纯 tensor 形式的 `clamp_min(1e-6)`，逻辑不变，但更稳定也更高效。

## 2. 当前 dipole pipeline 的整体结构

这套实现本质上是把 `flow_dagger` 的环境 / viewer / 人类干预框架保留下来，然后把策略训练部分替换成 discriminator-guided DIPOLE。

主流程入口是：

- `robosuite/pipeline/train_dipole.py`

核心模块分工是：

- `robosuite/pipeline/algorithms/dipole/agent.py`
  负责把 replay buffer、模型、checkpoint、normalizer 等组织成一个算法对象。

- `robosuite/pipeline/algorithms/dipole/models/flow.py`
  负责 dipole 双头 flow policy 本体，也就是：
  - `flow_head_pos`
  - `flow_head_neg`
  - 推理时的 guidance 组合
  - 训练时的双分支 weighted loss

- `robosuite/pipeline/algorithms/dipole/replay_buffer.py`
  负责：
  - 存 transition
  - 打 dipole label
  - 判断哪些序列可以用于训练
  - 采样 action chunk

- `robosuite/pipeline/algorithms/dipole/trainer.py`
  负责 learner update、异步训练线程、publish 节奏控制。

- `robosuite/pipeline/algorithms/dipole/workers.py`
  负责两个工作线程：
  - `DipolePolicyWorker`
  - `DipoleDiscriminatorWorker`

## 3. 训练时的数据流

### 3.1 初始化阶段

`train_dipole.py` 启动后，关键初始化顺序是：

1. 读取你指定的 flow multitask checkpoint
2. 从 checkpoint 推断 camera names / task metadata / normalizer
3. 按照 `flow_dagger` 的方式构建 robosuite env
4. 用 offline demos 初始化 demo buffer
5. 用 offline demos 拟合 action / proprio normalizer
6. 从 multitask flow checkpoint 初始化 dipole 双头模型
   这里的做法不是随机初始化两个 head，而是：
   - `flow_head_pos` 直接加载 base flow policy head
   - `flow_head_neg` 也用同一个 base flow head 拷贝初始化

这个初始化策略是合理的，因为它让正负两个分支在一开始都站在同一个预训练 policy 上，然后再被不同的权重逐渐拉开。

### 3.2 rollout 阶段

在线 rollout 循环仍然是 `flow_dagger` 风格：

1. 主线程维护环境交互
2. `policy_worker` 在线程里做策略推理
3. 如果有人类 intervention，则覆盖 policy action
4. 环境执行一步，得到 `next_obs`
5. transition 写入 online buffer
6. 如果是 intervention transition，也会额外写入 demo buffer
7. discriminator worker 异步消费这条 transition，对 prefix 做评估并输出 lambda / threshold / prediction
8. learner 用 demo + online 的混合 batch 更新 dipole policy
9. 训练到一定步数后发布新的 inference policy

## 4. discriminator 是怎么接到 dipole 里的

### 4.1 你的目标公式

你要求的是：

\[
w(s, a) = \sigma(\beta G(s, a)), \quad G(s, a)=\lambda
\]

其中：

- `lambda` 来自 online discriminator
- `lambda` 越高，表示越像 failure / OOD

### 4.2 代码里的实际映射

在 `DipoleFlowPolicy._compute_branch_weights()` 中，当前实现采用的是：

- `negative_weight = sigmoid(beta * lambda)`
- `positive_weight = 1 - negative_weight`
- 如果样本是 `force_positive`，则：
  - `positive_weight = 1`
  - `negative_weight = 0`

这里的语义是：

- **高 lambda 样本更偏向负分支**
- **低 lambda 样本更偏向正分支**
- **offline demo / intervention demo 被强制看成正样本**

这是我这次保留下来的核心假设，因为它和你的 failure discriminator 语义是对齐的：

- 成功 / 可信样本应该强化正分支
- 失败 / OOD 样本应该强化负分支

## 5. replay buffer 的关键逻辑

### 5.1 为什么不是所有 transition 都能立刻训练

`DipoleReplayBuffer` 不是简单地按单步采样，而是采样长度为 `action_horizon` 的连续 action chunk。

一个序列能成为 valid start，需要满足：

1. 从 `start` 到 `start + action_horizon - 1` 是连续 episode step
2. 中间不能提早 `done`
3. **起始 transition 必须已经 ready for dipole**

这里的 “ready” 指的是：

- 要么它是 `force_positive`
- 要么它已经拿到了 discriminator label

这意味着在线 non-intervention 数据不会在刚加入 buffer 的那一刻就参与训练，而是要等 discriminator 异步回填完 label 之后，才会进入 valid sequence 集合。

### 5.2 为什么只要求起始 transition ready

当前实现里，序列合法性只要求第一个 transition label ready。
这背后的含义是：这条 chunk 的权重是由 chunk 起点的 `lambda` 决定的，而不是每一个 step 各自一个权重。

这是一个明确的实现选择，不是 bug。
它和当前 batch 结构是配套的，因为 batch 里存的是：

- `action_sequences`: `[B, H, A]`
- `lambda_values`: `[B, 1]`

也就是说，一整个 chunk 只对应一个 `lambda`。

## 6. dipole 模型内部是怎么训练的

### 6.1 模型结构

`DipoleFlowModel` 是在 flow-multi 的 backbone 上做的双头扩展：

- 共享：
  - image encoder
  - proprio tokenizer
  - language encoder
  - fusion
  - condition aggregator

- 不共享：
  - `flow_head_pos`
  - `flow_head_neg`

也就是说，两个分支共享感知与条件建模，只在 action flow head 上分叉。

### 6.2 训练目标

对一个 batch，先构造：

- `noise`
- `t ~ Uniform(0, 1)`
- `x_t = (1 - t) * noise + t * action_sequence`
- `v_target = action_sequence - noise`

然后分别预测：

- `v_pos`
- `v_neg`

两个分支都计算三项 loss：

1. `flow_loss`
   让预测速度场逼近 `v_target`

2. `endpoint_loss`
   用 Euler 风格终点重建

3. `smooth_loss`
   约束预测出的 action chunk 在时间上更平滑

最后：

- 正分支 loss 乘 `positive_weight`
- 负分支 loss 乘 `negative_weight`
- 再按 `positive_loss_scale` / `negative_loss_scale` 组合成总 loss

### 6.3 为什么这和 flow_dagger 不同

`flow_dagger` 只有一个 flow head，本质上是行为克隆风格的 flow matching。

`dipole` 这里是把 policy 拆成：

- 倾向成功 / in-distribution 的正分支
- 倾向失败 / OOD 的负分支

所以训练时要同时学两个 head，推理时再做 guidance 组合。

## 7. 推理时是怎么出 action 的

推理函数入口在：

- `DipoleFlowPolicy.select_action()`

具体步骤是：

1. 把当前 observation 里的多路图像做 crop / resize / normalize
2. 把 proprio 用 checkpoint normalizer 标准化
3. 采样一个长度为 `action_horizon` 的初始噪声 chunk
4. 做 `n_ode_steps` 次积分

每一步的速度场使用：

\[
v = (1 + \omega) v_{pos} - \omega v_{neg}
\]

其中：

- `omega = guidance_scale`
- `v_pos` 是正头输出
- `v_neg` 是负头输出

然后得到一个完整的 action chunk，再按 `execute_horizon` 只执行前几步。

如果发生 intervention，会调用 `notify_intervention()` 把当前 chunk 清掉，避免人类接管后模型还沿用旧 chunk。

## 8. 这次优化为什么有效

### 8.1 原来的瓶颈

不管是训练还是推理，只要 observation 没变，多模态条件就没变。

但原来代码会在每个 ODE step 里重复做：

- 图像编码
- 语言编码
- token fusion
- condition aggregation

这在双头 dipole 里成本尤其高，因为本来已经比单头 flow 多一个 head。

### 8.2 现在的优化

现在流程变成：

1. 对当前 observation 编码一次 context
2. 之后所有 ODE steps 只复用这个 context
3. `pos/neg` head 直接用这个共享 context 出速度场

这不会改变数学逻辑，只是去掉了重复计算。

## 9. 这次修复后，我对当前实现的判断

### 9.1 已经比较稳定的部分

- env / viewer / rollout reset 主逻辑基本沿用了 `flow_dagger`
- multitask flow checkpoint -> dipole 双头初始化链路是通的
- discriminator worker -> online label 回填 -> online buffer 生效 这条链路结构上是完整的
- 异步 publish inference policy 的机制也是通的

### 9.2 仍然是实现假设的部分

当前实现有一个很重要的建模假设：

- **一个 action chunk 只用 chunk 起点的一个 lambda 来决定正负权重**

这不是错，但它是一个你后面可能还会继续迭代的设计点。
如果你之后想让 chunk 内每一步都有自己的 discriminator 信号，那么 batch 结构、replay buffer 和 loss 都需要一起改。

## 10. 如何运行

当前已有入口脚本：

- `robosuite/pipeline/scripts/train_dipole.sh`

最直接的运行方式是：

```bash
bash robosuite/pipeline/scripts/train_dipole.sh
```

如果你要显式指定训练模式：

```bash
DIPOLE_MODE=train bash robosuite/pipeline/scripts/train_dipole.sh
```

## 11. 这次改动对应的文件

- `robosuite/pipeline/algorithms/dipole/models/flow.py`
  - 复用多模态 context
  - 去掉 `_weighted_mean` 的 CPU 同步
  - 补了关键语义注释

- `robosuite/pipeline/train_dipole.py`
  - 增加 `online_buffer` 的恢复兼容性检查

