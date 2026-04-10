from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from robosuite.discriminator.utils.types import DetectorCalibrationSummary, TrajectoryDetectionResult

from .dataset import (
    LatentTrajectory,
    is_background_data_type,
    is_positive_data_type,
)
from .representation import FrozenTransitionRepresentation, TransitionFeatureSequence
from .support import SupportPenaltyCalibration, calibrate_support_penalty


def _resolve_device(device: str) -> torch.device:
    if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _torch_load_checkpoint(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass(frozen=True)
class PUHeadConfig:
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 1024
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    positive_sampling_ratio: float = 0.5
    grad_clip_norm: float = 1.0
    log_every: int = 100
    device: str = "cuda"


@dataclass(frozen=True)
class DetectorConfig:
    delta: float = 10.0
    lambda_mode: str = "mean"
    lambda_window_size: int = 12
    support_penalty_weight: float = 0.0
    c_min: float = 1e-3
    corrected_prob_eps: float = 1e-5


@dataclass(frozen=True)
class CalibrationTrajectory:
    pu_corrected_score: np.ndarray
    support_penalty: np.ndarray


@dataclass(frozen=True)
class PUCalibrationState:
    threshold: float
    c_estimate: float
    background_positive_rate: float
    support: SupportPenaltyCalibration
    delta: float
    lambda_mode: str
    lambda_window_size: int
    default_support_penalty_weight: float
    corrected_prob_eps: float
    positive_calibration_trajectories: list[CalibrationTrajectory]

    def to_payload(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "c_estimate": float(self.c_estimate),
            "background_positive_rate": float(self.background_positive_rate),
            "support_mean": float(self.support.mean),
            "support_std": float(self.support.std),
            "delta": float(self.delta),
            "lambda_mode": str(self.lambda_mode),
            "lambda_window_size": int(self.lambda_window_size),
            "default_support_penalty_weight": float(self.default_support_penalty_weight),
            "corrected_prob_eps": float(self.corrected_prob_eps),
            "positive_calibration_trajectories": [
                {
                    "pu_corrected_score": np.asarray(item.pu_corrected_score, dtype=np.float32),
                    "support_penalty": np.asarray(item.support_penalty, dtype=np.float32),
                }
                for item in self.positive_calibration_trajectories
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PUCalibrationState":
        return cls(
            threshold=float(payload["threshold"]),
            c_estimate=float(payload["c_estimate"]),
            background_positive_rate=float(payload.get("background_positive_rate", 0.0)),
            support=SupportPenaltyCalibration(
                mean=float(payload.get("support_mean", 0.0)),
                std=float(payload.get("support_std", 1.0)),
            ),
            delta=float(payload["delta"]),
            lambda_mode=str(payload["lambda_mode"]),
            lambda_window_size=int(payload["lambda_window_size"]),
            default_support_penalty_weight=float(payload.get("default_support_penalty_weight", 0.0)),
            corrected_prob_eps=float(payload.get("corrected_prob_eps", 1e-5)),
            positive_calibration_trajectories=[
                CalibrationTrajectory(
                    pu_corrected_score=np.asarray(item["pu_corrected_score"], dtype=np.float32),
                    support_penalty=np.asarray(item["support_penalty"], dtype=np.float32),
                )
                for item in payload.get("positive_calibration_trajectories", [])
            ],
        )


class _FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class PUClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        num_layers = max(1, int(num_layers))
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _aggregate_lambda(
    values: np.ndarray,
    *,
    lambda_mode: str,
    lambda_window_size: int,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size <= 0:
        return vals.astype(np.float32)
    mode = str(lambda_mode)
    if mode not in {"mean", "max"}:
        raise ValueError(f"Unsupported lambda_mode: {mode}")

    n = vals.shape[0]
    window = int(lambda_window_size)
    full_prefix = window <= 0

    if mode == "mean":
        if full_prefix:
            csum = np.cumsum(vals, dtype=np.float64)
            denom = np.arange(1, n + 1, dtype=np.float64)
            return (csum / denom).astype(np.float32)
        csum = np.cumsum(vals, dtype=np.float64)
        idx = np.arange(n, dtype=np.int64)
        start = np.maximum(0, idx - window + 1)
        left = np.where(start > 0, csum[start - 1], 0.0)
        denom = (idx - start + 1).astype(np.float64)
        return ((csum - left) / denom).astype(np.float32)

    if full_prefix:
        return np.maximum.accumulate(vals)

    out = np.empty(n, dtype=np.float32)
    dq: deque[int] = deque()
    for idx in range(n):
        while dq and dq[0] <= idx - window:
            dq.popleft()
        while dq and vals[dq[-1]] <= vals[idx]:
            dq.pop()
        dq.append(idx)
        out[idx] = float(vals[dq[0]])
    return out


def _compute_threshold(values: np.ndarray, delta: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= 0:
        raise ValueError("Cannot compute threshold from empty calibration values.")
    q = 100.0 * (1.0 - float(delta) / 100.0)
    return float(np.percentile(arr.astype(np.float64), q=q))


def _to_logit(prob: np.ndarray, eps: float) -> np.ndarray:
    clipped = np.clip(np.asarray(prob, dtype=np.float32), float(eps), 1.0 - float(eps))
    return (np.log(clipped) - np.log1p(-clipped)).astype(np.float32, copy=False)


def _flatten_sequences(sequences: Sequence[TransitionFeatureSequence]) -> tuple[np.ndarray, np.ndarray]:
    feature_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    for seq in sequences:
        if seq.features.shape[0] <= 0:
            continue
        feature_list.append(np.asarray(seq.features, dtype=np.float32))
        label = 1.0 if is_positive_data_type(seq.data_type) else 0.0
        label_list.append(np.full((seq.features.shape[0],), label, dtype=np.float32))
    if not feature_list:
        raise RuntimeError("Expected non-empty transition features for PU training.")
    return (
        np.concatenate(feature_list, axis=0).astype(np.float32),
        np.concatenate(label_list, axis=0).astype(np.float32),
    )


def _build_loader(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: PUHeadConfig,
) -> DataLoader:
    dataset = _FeatureDataset(features=features, labels=labels)
    num_positive = int(np.sum(labels > 0.5))
    num_background = int(labels.shape[0] - num_positive)
    use_balanced = num_positive > 0 and num_background > 0
    if use_balanced:
        ratio = min(max(float(cfg.positive_sampling_ratio), 0.0), 1.0)
        pos_weight = ratio / float(num_positive)
        bg_weight = (1.0 - ratio) / float(num_background)
        weights = torch.tensor(
            [pos_weight if float(label) > 0.5 else bg_weight for label in labels.tolist()],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            sampler=sampler,
            num_workers=int(cfg.num_workers),
            pin_memory=torch.cuda.is_available(),
        )
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def train_pu_classifier(
    *,
    train_sequences: Sequence[TransitionFeatureSequence],
    val_sequences: Sequence[TransitionFeatureSequence],
    cfg: PUHeadConfig,
) -> tuple[PUClassifierHead, dict[str, dict[str, float]]]:
    train_features, train_labels = _flatten_sequences(train_sequences)
    val_features, val_labels = _flatten_sequences(val_sequences)

    device = _resolve_device(cfg.device)
    model = PUClassifierHead(
        input_dim=int(train_features.shape[1]),
        hidden_dim=int(cfg.hidden_dim),
        num_layers=int(cfg.num_layers),
        dropout=float(cfg.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    train_loader = _build_loader(train_features, train_labels, cfg)
    val_loader = DataLoader(
        _FeatureDataset(features=val_features, labels=val_labels),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    history: dict[str, dict[str, float]] = {}

    def _run_epoch(loader: DataLoader, *, train: bool, epoch: int) -> dict[str, float]:
        model.train(mode=train)
        metrics: list[dict[str, float]] = []
        for step, batch in enumerate(loader):
            feature_b, label_b = batch
            feature_b = feature_b.to(device, non_blocking=True)
            label_b = label_b.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits = model(feature_b)
                loss = F.binary_cross_entropy_with_logits(logits, label_b)
                prob = torch.sigmoid(logits)
                pred = (prob >= 0.5).to(dtype=label_b.dtype)
                accuracy = torch.mean((pred == label_b).to(dtype=torch.float32))
                if train:
                    loss.backward()
                    if float(cfg.grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                    optimizer.step()
            stats = {
                "loss": float(loss.detach().item()),
                "accuracy": float(accuracy.detach().item()),
                "prob_mean": float(prob.detach().mean().item()),
            }
            metrics.append(stats)
            if int(cfg.log_every) > 0 and step % int(cfg.log_every) == 0:
                phase = "train" if train else "valid"
                print(
                    f"[lpb_dice {phase}] epoch={epoch:03d} step={step:05d} "
                    f"loss={stats['loss']:.6f} accuracy={stats['accuracy']:.4f}"
                )
        if not metrics:
            return {"loss": float("nan"), "accuracy": float("nan"), "prob_mean": float("nan")}
        return {
            key: float(np.mean([item[key] for item in metrics]))
            for key in metrics[0].keys()
        }

    for epoch in range(1, int(cfg.epochs) + 1):
        train_stats = _run_epoch(train_loader, train=True, epoch=epoch)
        val_stats = _run_epoch(val_loader, train=False, epoch=epoch)
        history[f"epoch_{epoch:03d}"] = {
            "train": train_stats,
            "valid": val_stats,
        }
        print(f"[lpb_dice detector] epoch={epoch:03d} train={train_stats} valid={val_stats}")

    model.eval()
    return model, history


def calibrate_pu_detector(
    *,
    model: PUClassifierHead,
    positive_sequences: Sequence[TransitionFeatureSequence],
    background_sequences: Sequence[TransitionFeatureSequence],
    cfg: DetectorConfig,
    device: str,
) -> PUCalibrationState:
    if not positive_sequences:
        raise RuntimeError("Positive calibration sequences cannot be empty.")

    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    @torch.no_grad()
    def _sequence_logits(seq: TransitionFeatureSequence) -> np.ndarray:
        feature_t = torch.from_numpy(np.asarray(seq.features, dtype=np.float32)).to(resolved_device)
        logits: list[torch.Tensor] = []
        batch_size = 4096
        for start in range(0, int(feature_t.shape[0]), batch_size):
            end = min(start + batch_size, int(feature_t.shape[0]))
            logits.append(model(feature_t[start:end]).detach().cpu())
        return torch.cat(logits, dim=0).numpy().astype(np.float32)

    positive_probs: list[np.ndarray] = []
    positive_support_raw: list[np.ndarray] = []
    corrected_positive_sequences: list[CalibrationTrajectory] = []

    for seq in positive_sequences:
        logits = _sequence_logits(seq)
        probs = torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32)
        positive_probs.append(probs)
        positive_support_raw.append(np.asarray(seq.support_penalty_raw, dtype=np.float32))

    c_estimate = float(np.mean(np.concatenate(positive_probs, axis=0)))
    c_estimate = max(c_estimate, float(cfg.c_min))
    support_calibration = calibrate_support_penalty(positive_support_raw)

    background_positive_rate = 0.0
    if background_sequences:
        bg_corrected_probs: list[np.ndarray] = []
        for seq in background_sequences:
            logits = _sequence_logits(seq)
            raw_prob = torch.sigmoid(torch.from_numpy(logits)).numpy().astype(np.float32)
            corrected_prob = np.clip(raw_prob / float(c_estimate), 0.0, 1.0)
            bg_corrected_probs.append(corrected_prob.astype(np.float32))
        if bg_corrected_probs:
            background_positive_rate = float(np.mean(np.concatenate(bg_corrected_probs, axis=0)))

    lambda_values: list[np.ndarray] = []
    for seq, raw_prob in zip(positive_sequences, positive_probs):
        corrected_prob = np.clip(raw_prob / float(c_estimate), 0.0, 1.0)
        pu_corrected_score = _to_logit(corrected_prob, eps=float(cfg.corrected_prob_eps))
        support_penalty = support_calibration.normalize(seq.support_penalty_raw)
        final_step_score = (
            -pu_corrected_score
            + float(cfg.support_penalty_weight) * support_penalty
        ).astype(np.float32)
        corrected_positive_sequences.append(
            CalibrationTrajectory(
                pu_corrected_score=pu_corrected_score,
                support_penalty=support_penalty,
            )
        )
        lambda_values.append(
            _aggregate_lambda(
                final_step_score,
                lambda_mode=str(cfg.lambda_mode),
                lambda_window_size=int(cfg.lambda_window_size),
            )
        )

    threshold = _compute_threshold(
        np.concatenate(lambda_values, axis=0).astype(np.float32),
        delta=float(cfg.delta),
    )
    return PUCalibrationState(
        threshold=float(threshold),
        c_estimate=float(c_estimate),
        background_positive_rate=float(background_positive_rate),
        support=support_calibration,
        delta=float(cfg.delta),
        lambda_mode=str(cfg.lambda_mode),
        lambda_window_size=int(cfg.lambda_window_size),
        default_support_penalty_weight=float(cfg.support_penalty_weight),
        corrected_prob_eps=float(cfg.corrected_prob_eps),
        positive_calibration_trajectories=corrected_positive_sequences,
    )


class LPBDiceDiscriminator:
    def __init__(
        self,
        *,
        representation: FrozenTransitionRepresentation,
        head: PUClassifierHead,
        calibration: PUCalibrationState,
        device: str = "cuda",
    ) -> None:
        self.representation = representation
        self.head = head
        self.calibration = calibration
        self.device = _resolve_device(device)
        self.action_horizon = int(self.representation.action_horizon)
        self.head.to(self.device)
        self.head.eval()
        self._threshold_cache: dict[float, float] = {
            float(self.calibration.default_support_penalty_weight): float(self.calibration.threshold)
        }

    @property
    def name(self) -> str:
        return "lpb_dice_pu"

    def close(self) -> None:
        return None

    def _threshold_for_weight(self, support_penalty_weight: float) -> float:
        weight = float(support_penalty_weight)
        if weight in self._threshold_cache:
            return float(self._threshold_cache[weight])
        lambda_values: list[np.ndarray] = []
        for seq in self.calibration.positive_calibration_trajectories:
            final_step_score = (
                -np.asarray(seq.pu_corrected_score, dtype=np.float32)
                + weight * np.asarray(seq.support_penalty, dtype=np.float32)
            ).astype(np.float32)
            lambda_values.append(
                _aggregate_lambda(
                    final_step_score,
                    lambda_mode=self.calibration.lambda_mode,
                    lambda_window_size=self.calibration.lambda_window_size,
                )
            )
        threshold = _compute_threshold(
            np.concatenate(lambda_values, axis=0).astype(np.float32),
            delta=self.calibration.delta,
        )
        self._threshold_cache[weight] = float(threshold)
        return float(threshold)

    @torch.no_grad()
    def _sequence_logits(self, sequence: TransitionFeatureSequence) -> np.ndarray:
        features = torch.from_numpy(np.asarray(sequence.features, dtype=np.float32)).to(self.device)
        logits: list[torch.Tensor] = []
        batch_size = 4096
        for start in range(0, int(features.shape[0]), batch_size):
            end = min(start + batch_size, int(features.shape[0]))
            logits.append(self.head(features[start:end]).detach().cpu())
        return torch.cat(logits, dim=0).numpy().astype(np.float32)

    def detect_trajectory(
        self,
        trajectory: LatentTrajectory,
        *,
        labels: Optional[np.ndarray] = None,
        adaptive_threshold: bool = False,
        delta_min: float = 0.0,
        delta_max: float = 100.0,
        warmup_steps: int = 0,
        update_interval: int = 1,
        support_penalty_weight: Optional[float] = None,
    ) -> TrajectoryDetectionResult:
        seq = self.representation.encode_trajectory(trajectory)
        raw_logits = self._sequence_logits(seq)
        raw_prob = torch.sigmoid(torch.from_numpy(raw_logits)).numpy().astype(np.float32)
        corrected_prob = np.clip(raw_prob / float(self.calibration.c_estimate), 0.0, 1.0)
        pu_corrected_score = _to_logit(corrected_prob, eps=self.calibration.corrected_prob_eps)
        support_penalty = self.calibration.support.normalize(seq.support_penalty_raw)
        weight = (
            float(self.calibration.default_support_penalty_weight)
            if support_penalty_weight is None
            else float(support_penalty_weight)
        )
        final_step_score = (
            -pu_corrected_score
            + weight * support_penalty
        ).astype(np.float32)
        aggregate_scores = _aggregate_lambda(
            final_step_score,
            lambda_mode=self.calibration.lambda_mode,
            lambda_window_size=self.calibration.lambda_window_size,
        )
        threshold = float(self._threshold_for_weight(weight))
        thresholds = np.full_like(aggregate_scores, threshold, dtype=np.float32)
        predictions = (aggregate_scores >= threshold).astype(np.int64)
        first_crossing = np.where(predictions == 1)[0]
        first_crossing_index = int(first_crossing[0]) if first_crossing.size > 0 else None
        return TrajectoryDetectionResult(
            detector_name=self.name,
            step_scores=final_step_score.astype(np.float32),
            aggregate_scores=aggregate_scores.astype(np.float32),
            thresholds=thresholds.astype(np.float32),
            predictions=predictions.astype(np.int64),
            aux_scores=support_penalty.astype(np.float32),
            labels=None,
            metadata={
                "pu_raw_score": raw_logits.astype(np.float32),
                "pu_corrected_score": pu_corrected_score.astype(np.float32),
                "support_penalty": support_penalty.astype(np.float32),
                "final_step_score": final_step_score.astype(np.float32),
                "threshold_final": float(threshold),
                "c_estimate": float(self.calibration.c_estimate),
                "background_positive_rate": float(self.calibration.background_positive_rate),
                "support_penalty_weight": float(weight),
                "support_penalty_mean": float(self.calibration.support.mean),
                "support_penalty_std": float(self.calibration.support.std),
                "delta_final": float(self.calibration.delta),
                "first_crossing_index": first_crossing_index,
                "pu_raw_score_mean": float(np.mean(raw_logits)),
                "pu_corrected_score_mean": float(np.mean(pu_corrected_score)),
                "support_penalty_mean_sequence": float(np.mean(support_penalty)),
                "final_step_score_mean": float(np.mean(final_step_score)),
            },
        )

    def calibration_summary(self) -> DetectorCalibrationSummary:
        return DetectorCalibrationSummary(
            detector_name=self.name,
            threshold=float(self.calibration.threshold),
            metadata={
                "c_estimate": float(self.calibration.c_estimate),
                "background_positive_rate": float(self.calibration.background_positive_rate),
                "support_penalty_weight": float(self.calibration.default_support_penalty_weight),
                "support_penalty_mean": float(self.calibration.support.mean),
                "support_penalty_std": float(self.calibration.support.std),
                "delta": float(self.calibration.delta),
                "lambda_mode": str(self.calibration.lambda_mode),
                "lambda_window_size": int(self.calibration.lambda_window_size),
            },
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        device: str = "cuda",
    ) -> "LPBDiceDiscriminator":
        payload = _torch_load_checkpoint(checkpoint_path, map_location="cpu")
        if str(payload.get("format_version", "")) != "lpb_dice_v1":
            raise RuntimeError(
                "Expected an lpb_dice checkpoint with format_version=lpb_dice_v1, "
                f"got keys={sorted(payload.keys())}"
            )
        representation = FrozenTransitionRepresentation.from_backbone_payload(
            payload=payload["backbone"],
            device=device,
            batch_size=int(payload["representation"].get("batch_size", 256)),
            source_ckpt=payload["backbone"].get("source_ckpt", None),
        )
        head_cfg = payload["detector"]["head_config"]
        head = PUClassifierHead(
            input_dim=int(payload["detector"]["feature_dim"]),
            hidden_dim=int(head_cfg["hidden_dim"]),
            num_layers=int(head_cfg["num_layers"]),
            dropout=float(head_cfg["dropout"]),
        )
        head.load_state_dict(payload["detector"]["head_state"], strict=True)
        calibration = PUCalibrationState.from_payload(payload["detector"]["calibration"])
        return cls(
            representation=representation,
            head=head,
            calibration=calibration,
            device=device,
        )
