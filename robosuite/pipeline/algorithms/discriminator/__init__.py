"""Online-trainable BCE discriminator + shared frozen LPB v2 encoder.

Skeleton — see robosuite/pipeline/docs/DIPOLE_RL.md and
robosuite/pipeline/docs/prompts/02_online_discriminator.md.
"""

from .base import (
    DiscriminatorBase,
    DiscriminatorBatch,
    DiscriminatorOutput,
)
from .encoder import SharedFrozenEncoder
from .online_bce import DiscriminatorConfig, OnlineBCEDiscriminator
from .replay import DiscriminatorReplayBuffer

__all__ = [
    "DiscriminatorBase",
    "DiscriminatorOutput",
    "DiscriminatorBatch",
    "DiscriminatorConfig",
    "OnlineBCEDiscriminator",
    "DiscriminatorReplayBuffer",
    "SharedFrozenEncoder",
]
