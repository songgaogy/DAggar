from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class NetworkConfig:
    visual_dim: int = 3 * 9 * 256
    proprio_dim: int = 14
    action_horizon: int = 8
    action_dim: int = 7
    hidden_dims: tuple[int, ...] = (1024, 1024, 1024)
    latent_limit: float = 2.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @property
    def chunk_dim(self) -> int:
        return self.action_horizon * self.action_dim

    @property
    def state_dim(self) -> int:
        """Width of the flattened frozen visual tokens and proprioception."""
        return self.visual_dim + self.proprio_dim

    def validate(self) -> None:
        integer_fields = (
            self.visual_dim,
            self.proprio_dim,
            self.action_horizon,
            self.action_dim,
        )
        if any(value <= 0 for value in integer_fields[:1] + integer_fields[2:]):
            raise ValueError("Network dimensions must be positive.")
        if self.proprio_dim < 0:
            raise ValueError("proprio_dim must be non-negative.")
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive values.")
        if self.latent_limit <= 0:
            raise ValueError("latent_limit must be positive.")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max.")


@dataclass(frozen=True)
class DSRLConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    learner_device: str = "cuda:0"
    learning_rate: float = 3e-4
    gamma: float = 0.97
    tau: float = 0.005
    batch_size: int = 256
    utd_steps: int = 30
    target_entropy: float = 0.0
    initial_alpha: float = 1.0
    grad_clip_norm: float | None = None

    def validate(self) -> None:
        self.network.validate()
        if not self.learner_device.startswith("cuda:"):
            raise ValueError("DSRL requires an explicit CUDA device such as 'cuda:0'.")
        if self.learning_rate <= 0 or self.batch_size <= 0:
            raise ValueError("learning_rate and batch_size must be positive.")
        if self.utd_steps <= 0:
            raise ValueError("utd_steps must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        if self.initial_alpha <= 0:
            raise ValueError("initial_alpha must be positive.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when provided.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
