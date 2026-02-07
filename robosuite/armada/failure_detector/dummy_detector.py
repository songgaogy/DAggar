import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List


class DummyFailureDetector:
    """
    A dummy failure detector for testing the RobosuiteRunner pipeline.
    It triggers a fake failure at a specific timestep and supports controlled rewinding.
    """
    def __init__(
            self, 
            trigger_step: int = 200,
            rewind_steps: int = 16,
            **kwargs
        ) -> None:
        """
        Args:
            trigger_step: The timestep to trigger a simulated failure.
            rewind_steps: How many steps to rewind when intervention starts.
        """
        self.trigger_step = trigger_step
        self.rewind_steps = rewind_steps
        
        # State
        self.has_triggered = False
        self.max_episode_length = 1000
        self.Ta = 8 # Default, will be updated in runtime_initialize
        self.device = None

    def runtime_initialize(self, 
                           device: torch.device, 
                           policy: torch.nn.Module,
                           replay_buffer: Any,
                           episode_manager: Any,
                           max_episode_length: int
                           ) -> None:
        """
        Initialize runtime variables passed from the runner.
        """
        self.device = device
        self.max_episode_length = max_episode_length
        if hasattr(episode_manager, 'Ta'):
            self.Ta = episode_manager.Ta
            
        print(f"[DummyDetector] Initialized. Will trigger failure at step {self.trigger_step}.")

    def process_step(self, step_data: Dict[str, Any]) -> None:
        """
        Process a step (either episode_start or policy_step).
        """
        step_type = step_data['step_type']

        if step_type == 'episode_start':
            self.has_triggered = False
            print(f"[DummyDetector] Episode started. Reset trigger state.")

        elif step_type == 'policy_step':
            timestep = step_data['timestep']
            # Just log periodically
            if timestep % 10 == 0:
                print(f"[DummyDetector] Monitoring step {timestep}...")

    def detect_failure(self, timestep: int, max_episode_length: int) -> Tuple[bool, Optional[str], int]:
        """
        Check if failure condition is met.
        """
        # Trigger failure if we reached the step and haven't triggered yet
        if not self.has_triggered and timestep >= self.trigger_step:
            self.has_triggered = True
            print(f"[DummyDetector] !!! TRIGGERING DUMMY FAILURE AT STEP {timestep} !!!")
            return True, "Simulated Dummy Failure", timestep
        
        return False, None, timestep

    def wait_for_final_results(self, j: int) -> Tuple[bool, Optional[str], int]:
        """
        Sync method (no-op for dummy).
        """
        return self.detect_failure(j, self.max_episode_length)

    def should_stop_rewinding(self, *args, **kwargs) -> bool:
        """
        Method existence check used by RobosuiteRunner to determine rewind logic.
        Return True is not strictly used by runner's check logic (it checks hasattr),
        but logic inside rewind_step might use it.
        """
        return False

    def rewind_step(self, j: int, episode_buffers: Dict[str, List], curr_timestep: int) -> bool:
        """
        Control the rewinding process.
        Args:
            j: Current timestep during rewinding (decreasing).
            curr_timestep: The timestep where failure occurred.
        Returns:
            True to continue rewinding, False to stop.
        """
        target_step = max(0, curr_timestep - self.rewind_steps)
        
        # We only allow rewinding at chunk boundaries (Ta) to be safe/simple
        if j <= target_step:
            print(f"[DummyDetector] Rewind target reached [{target_step}]({curr_timestep} -> {target_step}). Stopping rewind.")
            return False
        
        return True

    def finalize_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return dummy failure indices.
        """
        # Create a dummy failure mask
        n_steps = episode['action_mode'].shape[0]
        failure_indices = np.zeros((n_steps,), dtype=np.bool_)
        
        # Mark failure if we triggered
        if self.has_triggered:
            # Mark the trigger step as failure point
            # Note: indices must match action_mode length
            idx = min(self.trigger_step, n_steps - 1)
            failure_indices[idx] = True
            print(f"[DummyDetector] Finalizing episode. Marked failure at index {idx}.")
            
        return {'failure_indices': failure_indices}

    def cleanup(self) -> None:
        print("[DummyDetector] Cleanup called.")