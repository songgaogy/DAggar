# D4-Disc Refactor — LPB-Architecture Parity + Modular Layout

## Context

Current `robosuite/discriminator/d4disc/` 把 policy 的 `flow_multi` encoder 当作冻结前端
(`FlowMultiEncoderWrapper`, 256-dim) + 读取 `data/.lpb_score_cache/<task>/<sha1>.npz`
里的预编码 latent。用户想让它改成**与原始 LPB 架构一致**（trainable ResNet-18 encoder +
transformer dynamics，但保留 d4disc 独有的 AdaLN / γ / Phase A–B / CFG），并且：

- 直接从 `data/.lpb_score_preprocessed_cache/<task>/<sha1>.npz` 读取**预解码的图像** —
  格式 `images_chw (T, 3, 3, 128, 128) uint8`, `proprio (T, 14) f32`, `actions (T, 7) f32`，
  只用 cam-0 (agentview) @ 原生 128×128。
- **不**读 raw HDF5 图像（太慢），不在训练时重新 decode。
- 当 `skip_bootstrap=True` 且 `ω=0` 时 ≡ LPB（**架构等价**：同一个 ResNet-18 encoder + 同一个
  transformer predictor；score = `‖f(+|z) − z_target‖² / (2σ²)`，对应 design doc §7.1 的
  "learned one-step likelihood as parametric analogue of LPB's KNN `−log p_+`"，不是 byte-level 等价）。
- 把当前扁平的 package 拆成 `models/ data/ training/ inference/` 4 个子包。

所有 d4disc 专属算法（γ buffer / advantage gate / KNN warm-start / AdaLN-Zero /
CFG ω-mix / collapse monitor / EMA / schedule / 每任务 conformal τ）**保持数值 byte-level 等价**；
改动只在 encoder + dataset + 目录。

## Target layout

```
robosuite/discriminator/d4disc/
├── __init__.py                 # public re-exports (API stable)
├── models/
│   ├── __init__.py
│   ├── encoder.py              # NEW: verbatim copy of lpb.model.Encoder (ResNet-18, 512-d)
│   ├── adaln.py                # MOVED from d4disc/adaln.py (unchanged)
│   └── dynamics.py             # MOVED from d4disc/model.py (ConditionalDynamicsPredictor)
├── data/
│   ├── __init__.py
│   ├── cache.py                # NEW: PreprocessedCacheReader (reads .npz only, no encoding)
│   └── dataset.py              # REWRITTEN: image-based; keeps gamma/sample_idx/is_fail_raw
├── training/
│   ├── __init__.py
│   ├── schedule.py             # MOVED (unchanged)
│   ├── ema.py                  # MOVED (unchanged)
│   ├── monitor.py              # MOVED (unchanged)
│   ├── filter.py               # MODIFIED: accept encoder; gate/warm-start encode on-device
│   └── trainer.py              # MODIFIED: forward = encode(img)→z; rest unchanged
├── inference/
│   ├── __init__.py
│   ├── feature.py              # RENAMED from dynamics_feature.py; uses Encoder+cache_reader
│   ├── detector.py             # MODIFIED: import paths only (scoring math unchanged)
│   └── benchmark.py            # RENAMED from d4_benchmark.py; new ctor args
├── tests/
│   ├── test_adaln.py           # import path updates only
│   ├── test_advantage_gate.py  # use _FakeEncoder to decouple from ResNet
│   ├── test_warm_start.py      # same pattern
│   └── test_cache_loader.py    # NEW
├── scripts/
│   ├── train_d4.sh             # drop POLICY_CKPT/CACHE_ROOT; add PREPROC_CACHE_ROOT/IMAGE_SIZE
│   ├── run_d4_benchmark.sh     # same
│   ├── sweep_omega_d4.sh       # same
│   ├── ablate_schedule.sh      # same
│   └── visualize_d4.sh         # same
├── train_d4.py                 # MODIFIED CLI: drop --policy-ckpt/--cache-root,
│                                              add --preprocessed-cache-root/--image-size/
│                                              --encoder-pretrained/--encoder-freeze/--encoder-checkpoint
└── visualize.py                # MODIFIED analogously
```

## Critical implementation notes

### Architectural parity with LPB (load-bearing for degeneration claim)

- `models/encoder.py` 是 `lpb/model.py::Encoder` 的 **verbatim copy**（ResNet-18，ImageNet
  pretrained，`freeze=True` 默认，`latent_dim=512`）。不 import LPB，避免 coupling。
- `models/dynamics.py::ConditionalDynamicsPredictor` 构造时传 `latent_dim=512`。其 transformer
  结构已经和 `lpb.model.DynamicsPredictor` 一致（6 层 8 头 d_model=512，`pred_latent_token` +
  `pred_proprio_token` learnable query，pos embedding），只是 decoder block 换成 AdaLN 版。
- **关键不变式**：`adaln_init_std=0` + `residual_latent_head=True` + `cond_idx=COND_PLUS`
  时，`AdaLNModulation` 全零 → `AdaLNDecoderBlock` 退化为 `x+attn(LN(x))` / `x+mlp(LN(x))`，
  等价 `lpb.model.DecoderBlock`；residual head 零初始化 → `pred_latent = z_t`。配合
  Phase A (c=+ only, clean positives only) 训练，就等价于 LPB 的训练 forward
  `f_θ(z, s, a) ≈ z_{t+h}` with `MSE(pred, enc(target_img).detach())`。
- 在 `skip_bootstrap + ω=0` 的推断路径下，score = `‖f(+|z) − z_target‖²`，就是 design doc
  §7.1 描述的 parametric LPB (not byte-identical to LPB's KNN, 是同 encoder / 同 feature space
  的 learned dynamics 变体)。

### Dataset (`data/dataset.py`)

- 不再继承 `d3disc.dataset.LatentFlowDynamicsDataset`（那是 flow_multi-latent 架构）。
  重新实现 scanning + indexing。
- `__init__(cache_reader, expert_paths, rollout_paths, fail_rollout_paths, horizon,
  proprio_indices, max_trajectories_per_kind, image_size=128)`。
- Scanning：遍历 HDF5 仅为了枚举 `(file_path, demo_key)` 和拿 trajectory length 与
  `proprio_dim/action_dim`；**每个 sample 的数据来自 `cache_reader.load(task, file, demo_key)`**。
- `__getitem__(idx) → dict`（**keys 与现版本保持向后兼容**）：
  ```
  current_image:   uint8 (3, 128, 128)     # cam-0 at t, float 转换在 encoder.forward 里
  current_proprio: f32 (proprio_dim,)
  action_sequence: f32 (H, action_dim)     # 末端 edge-pad
  target_image:    uint8 (3, 128, 128)     # cam-0 at t+h, 越界 edge-clamp
  target_proprio:  f32 (proprio_dim,)
  is_expert:       i64 (scalar)
  sample_idx:      i64 (scalar)
  is_fail_raw:     bool (scalar)
  gamma:           f32 (scalar)
  ```
- 保留 `gamma_buffer (N,) f32`（init 0.5）+ `update_gamma(indices, new_gamma, ema_alpha,
  clamp_eps)` + `fail_indices()` 接口（与当前 d4disc 完全一致）。
- 保留 `preload_preprocessed(self) -> int`（对应 d3disc 的 `preload_states_actions`，在
  DataLoader worker fork 前 warm 所有 demo tensor）。
- **Uint8 images kept until encoder**: 节省 ~4× CPU→GPU 带宽；`lpb.Encoder.forward` 已内置
  `if max>1.5: x = x/255.0`，不需要 dataset 侧转换。
- **Edge-clamp**: `th = min(t+h, length-1); t0 = max(0, th - h)`；action 不足则 edge-pad。
- 删除 `latent_dim` property（latents 不再由 dataset 产生）。
- 删除 `materialize_missing_latent_caches`（预处理 cache 外部生成，用户禁用自动构建）。

### Preprocessed cache reader (`data/cache.py`)

```python
@dataclass(frozen=True)
class PreprocessedDemo:
    images_chw: np.ndarray   # (T, 3, 128, 128) uint8  —— 已切 cam-0
    proprio:    np.ndarray   # (T, 14) f32
    actions:    np.ndarray   # (T, 7) f32
    length:     int

class PreprocessedCacheReader:
    def __init__(self, cache_root="data/.lpb_score_preprocessed_cache", *,
                 image_size=128, camera_index=0,
                 cache_key_fn: Callable[[str,str,str], str] | None = None): ...
    def cache_key(self, task, file_path, demo_key) -> str
    def cache_path(self, task, file_path, demo_key) -> str
    def load(self, task, file_path, demo_key) -> PreprocessedDemo  # miss → FileNotFoundError
```

- **默认 `cache_key_fn`**：沿用 `d3disc/encoder.py::FlowMultiEncoderWrapper.cache_key`
  配方 **去掉 `checkpoint_path` 字段**（因为预处理 cache 与 encoder 无关）。
  recipe：`"|".join([task, resolve(file_path), demo_key, image_size,
  ",".join(camera_names), mtime(file_path)])` → SHA1。
- ⚠️ **Open Question #1**（见下方）：这个配方**未经验证**与磁盘上 492 个 `<sha1>.npz`
  相匹配。外部生成工具的 Python 源不在 repo 里。若 mismatch，用户需给出正确 recipe 或
  让我切换到 `index.json` 方案。

### Encoder integration in training (`training/trainer.py`)

- Constructor: `D4Trainer(predictor, encoder, dataset, *, config, device)`。
- `D4TrainerConfig` 新增：`encoder_freeze: bool = True`（默认冻结）、
  `encoder_lr_mult: float = 0.1`（unfreeze 时用 0.1× predictor lr）。
- `_run_step` 新 forward：
  ```python
  enc_ctx = torch.no_grad() if self.encoder_frozen else contextlib.nullcontext()
  with enc_ctx:
      z_t = self.encoder(b["current_image"].to(device))                     # (B, 512)
  z_tph = self.encoder(b["target_image"].to(device)).detach()               # 永远 detach target
  out = self.predictor(z_t, b["current_proprio"].to(device),
                       b["action_sequence"].to(device), cond_idx=cond)
  loss_latent  = F.mse_loss(out["pred_latent"],  z_tph)
  loss_proprio = F.mse_loss(out["pred_proprio"], b["target_proprio"].to(device))
  ```
  与 LPB `trainer.py` 的 forward 完全同构，只多了 `cond_idx` 参数。
- `_eval_r_plus_on_held_out`、`_eval_conditional_delta`、`_cache_probe_batch`：相应地
  把 batch 里的 `current_latent/target_latent` 读取改为 encoder 前向。
- `build_payload`: 额外保存 `encoder_state_dict` + `encoder_config` 元数据（便于
  benchmark 重建；`freeze=True` 时可选不存，节省 44MB）。
- Phase A/B 调度、condition sampling、EMA 更新、collapse detector、grad clip、save cadence
  全部 **不改动**。

### Encoder integration in filter (`training/filter.py`)

- `compute_advantage_gate(predictor, encoder, dataset, *, alpha_k, kappa_k, sigma_sq,
  device, batch_size, num_workers, ema_alpha)`：在 DataLoader 循环里对 `current_image`
  / `target_image` 用 `encoder.eval()` + `no_grad()` 编码得到 `z_t` / `z_target`，再算
  `r+ = ‖f(+|z) − z_target‖²`、`r- = ‖f(-|z) − z_target‖²`、`A = (r- − r+)/(2σ²)`、
  `γ = σ(α · (−A − κ))`。每个 bootstrap outer epoch 对 fail 集 encode 一次
  (~5–10s @ 100k frames on 4090)。
- `warm_start_gamma_from_knn(dataset, encoder, *, k, mode, ..., device, batch_size,
  num_workers)`：`_gather_z(indices)` 用 encoder 现场编码 clean + fail bank，后续
  rank/sigmoid calibration 完全不变。复用 `d3disc.filter.knn_sqdist`（纯数学工具，
  无 flow_multi 依赖）。

### Inference (`inference/feature.py`, `detector.py`, `benchmark.py`)

- `D4FeatureExtractor(encoder, cache_reader, latent_dim=512, proprio_dim, action_dim,
  action_horizon, proprio_indices)`：`.extract(task, file, demo_key, states, actions,
  horizon)` → `D4Frames(z_current, z_target, proprio, action_chunks, length)`。
  内部 `demo = cache_reader.load(...)`；`imgs = demo.images_chw[:length]`
  （已切 cam-0）；`z_all = encoder(to_tensor(imgs).to(device))` 按
  `encoder_batch_size` 分批；`z_target` 用与训练相同的 edge-clamp helper 生成。
- `D4Frames` **dataclass 形状不变**（`z_current / z_target / proprio / action_chunks /
  length` 这五个字段），下游 `detector.py` / `benchmark.py` / `visualize.py` 无需改。
- `D4Detector` 仅改 import 路径（`.feature` / `..models.adaln` / `..models.dynamics`）；
  scoring 数学 byte-level 不变。
- `D4BenchmarkDiscriminator`：构造参数换成 `encoder_pretrained / encoder_freeze /
  preprocessed_cache_root / image_size`；内部 `self.encoder = Encoder(...)` +
  `self.cache_reader = PreprocessedCacheReader(...)` + `self.feature_extractor = D4FeatureExtractor(...)`。
  加载 ckpt 时先 `load_state_dict(encoder_state_dict, strict=True)` 再构造 detector；
  old 256-d flow_multi ckpt 直接 `raise` 给清晰错误（**not supported**；用户确认希望硬失败）。

### CLI / scripts

- `train_d4.py`：删除 `--policy-ckpt`、`--cache-root`；新增
  `--preprocessed-cache-root` (default `data/.lpb_score_preprocessed_cache`)、
  `--image-size` (default 128)、`--encoder-pretrained/--no-encoder-pretrained` (True)、
  `--encoder-freeze/--no-encoder-freeze` (True)、`--encoder-checkpoint` (optional 预训练
  ResNet-18 state_dict 路径)。构造顺序：`cache_reader → dataset → dataset.preload_preprocessed()
  → encoder → predictor(latent_dim=encoder.latent_dim) → trainer.run()`。
- `scripts/train_d4.sh`、`run_d4_benchmark.sh`、`sweep_omega_d4.sh`、
  `ablate_schedule.sh`、`visualize_d4.sh`：把 `POLICY_CKPT`/`CACHE_ROOT` 替换成
  `PREPROC_CACHE_ROOT`/`IMAGE_SIZE`，其它 env 变量全部保留。
- `visualize.py`：同 benchmark 的改造。

### Public API stability

`d4disc/__init__.py` 重新 export：
```python
from .models.encoder import Encoder
from .models.adaln import AdaLNDecoderBlock, AdaLNModulation, ConditionEmbedder
from .models.dynamics import ConditionalDynamicsPredictor
from .data.cache import PreprocessedCacheReader, PreprocessedDemo
from .data.dataset import LatentFlowDynamicsDatasetD4
from .training.schedule import D4Schedule
from .training.ema import ModelEMA
from .training.monitor import CollapseDetector, D4Health
from .training.filter import compute_advantage_gate, warm_start_gamma_from_knn
from .training.trainer import D4Trainer, D4TrainerConfig
from .inference.feature import D4FeatureExtractor, D4Frames
from .inference.detector import D4Detector, D4StepOutput
from .inference.benchmark import D4BenchmarkDiscriminator
```
外部调用 `from robosuite.discriminator.d4disc import D4Trainer` 等**保持可用**。

## Files to modify (index)

| 新路径 | 来源 | 性质 |
|---|---|---|
| `models/encoder.py` | `lpb/model.py::Encoder` | 新建（verbatim copy） |
| `models/adaln.py` | `d4disc/adaln.py` | 移动（不改） |
| `models/dynamics.py` | `d4disc/model.py` | 移动 + 改 import |
| `data/cache.py` | 新建 | 新建 |
| `data/dataset.py` | `d4disc/dataset.py` | 重写（图像 schema） |
| `training/schedule.py` | `d4disc/schedule.py` | 移动（不改） |
| `training/ema.py` | `d4disc/ema.py` | 移动（不改） |
| `training/monitor.py` | `d4disc/monitor.py` | 移动（不改） |
| `training/filter.py` | `d4disc/filter.py` | 改（encoder 参数 + 现场编码） |
| `training/trainer.py` | `d4disc/trainer.py` | 改（forward 改为 encoder-based） |
| `inference/feature.py` | `d4disc/dynamics_feature.py` | 重写 |
| `inference/detector.py` | `d4disc/detector.py` | 改 import 路径 |
| `inference/benchmark.py` | `d4disc/d4_benchmark.py` | 改构造签名 |
| `train_d4.py` | 同名 | 改 flags |
| `visualize.py` | 同名 | 改 flags |
| `scripts/*.sh` | 同名 | 改 env 变量 |
| `tests/test_adaln.py` | 同名 | 改 import |
| `tests/test_advantage_gate.py` | 同名 | 改 import + FakeEncoder |
| `tests/test_warm_start.py` | 同名 | 改 import + FakeEncoder |
| `tests/test_cache_loader.py` | 新建 | 新建 |
| `d4disc/__init__.py` | 同名 | 改 re-exports |

**删除旧文件**（移动后）：`adaln.py`, `model.py`, `dataset.py`, `schedule.py`, `ema.py`,
`monitor.py`, `filter.py`, `trainer.py`, `dynamics_feature.py`, `detector.py`, `d4_benchmark.py`。

## Verification

1. **Unit tests** (pytest, CPU ok):
   - `tests/test_adaln.py`：identity-at-init（不变）。
   - `tests/test_advantage_gate.py`：用 `_FakeEncoder(nn.Module)` 返回固定 latent，
     验证 γ 公式。
   - `tests/test_warm_start.py`：同上，验证 rank / sigmoid 两模式。
   - `tests/test_cache_loader.py`（新）：synth 一个 toy `.npz` 放到 tmp_path，注入
     `cache_key_fn`，验证 cam-0 切片、dtype、length；miss 抛 `FileNotFoundError`。
2. **Import smoke**:
   ```
   python -c "from robosuite.discriminator.d4disc import (
       D4Trainer, D4TrainerConfig, D4Detector, D4BenchmarkDiscriminator,
       ConditionalDynamicsPredictor, Encoder, PreprocessedCacheReader,
       LatentFlowDynamicsDatasetD4, compute_advantage_gate,
       warm_start_gamma_from_knn)"
   ```
3. **Architectural-parity spot check**（run by user）：构造 `adaln_init_std=0,
   residual_latent_head=True` 的 fresh predictor，喂一个 batch with `cond_idx=COND_PLUS`，
   断言 `pred_latent ≈ z_t` (MSE < 1e-6)。这是 "degenerates to LPB" 的核心不变式。
4. **Integration smoke**（**不要由我跑**，用户手动）：
   ```
   SKIP_BOOTSTRAP=1 WARM_UP_EPOCHS=2 FAIL_NUM=8 SUCC_NUM=16 EXPERT_NUM=0 \
   TASKS="PandaLift" bash robosuite/discriminator/d4disc/scripts/train_d4.sh
   OMEGA=0 bash robosuite/discriminator/d4disc/scripts/run_d4_benchmark.sh
   ```
   预期 AUROC 进入 LPB 的区间（非随机）→ confirms parametric-LPB degeneration。
5. **Checkpoint round-trip**: train 存下后 benchmark 加载，`arch_args["latent_dim"] == 512`
   且 `ema_state_dict` 能 `strict=True` 加载。旧 flow_multi ckpt (latent_dim=256) 显式
   报错。

## Open question (blocking)

**Preprocessed-cache SHA1 scheme unknown**。磁盘上有 492 个
`data/.lpb_score_preprocessed_cache/PandaLift/<sha1>.npz`，但 repo 里没有生成代码
（外部工具）。`PreprocessedCacheReader.cache_key` 默认我会用 d3disc 的 recipe
**去掉 checkpoint_path 字段**：
```python
"|".join([task, resolve(file_path), demo_key, image_size,
          ",".join(camera_names), mtime(file_path)])  # SHA1
```

但这是**未验证猜测**。要么：
- (a) 用户确认 recipe（最理想：一行文字告诉我哪些字段进 hash），
- (b) 我写一个一次性工具 rehash 现有 `.npz` 到 d4disc 自己的 recipe（user-owned build step），
- (c) 用户提供/我生成 `index.json: {sha1: {task, file_path, demo_key}}` 映射，
  `cache_key_fn` 走查表。

**在 open question 解决前不动手实现**。给出 recipe/方案后我即可开始。

## Out of scope

- 不动 `d3disc/` — 仍独立可用。
- 不动 `lpb/` — encoder 只是**复制**到 d4disc，不加 import 依赖。
- 不修 `monitor.py` 里 `branch_starvation` sign bug（`d4disc_v2_algorithm_design.md` §2
  已记录，属于算法层面后续事项；本次 refactor 不碰算法）。
- 不处理 multi-camera（用户已拍板单 agentview）。
- 不重建 preprocessed cache（用户负责）。
