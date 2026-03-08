from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NoopImageRunner(BaseImageRunner):
    """
    Runner used when no online rollout evaluation environment is available.
    """

    def run(self, policy: BaseImagePolicy) -> Dict:
        _ = policy
        return {}
