from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from robosuite.pipeline.utils.runtime import capture_rng_state, restore_rng_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_restore_rng_state_accepts_cuda_mapped_byte_tensors() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    expected = capture_rng_state()

    mapped = {
        "python": expected["python"],
        "numpy": expected["numpy"],
        "torch": expected["torch"].to("cuda:0"),
        "cuda": [value.to("cuda:0") for value in expected["cuda"]],
    }

    restore_rng_state(mapped)

    restored = capture_rng_state()
    assert restored["python"] == expected["python"]
    assert restored["numpy"][0] == expected["numpy"][0]
    assert np.array_equal(restored["numpy"][1], expected["numpy"][1])
    assert restored["numpy"][2:] == expected["numpy"][2:]
    assert restored["torch"].numpy().tobytes() == expected["torch"].numpy().tobytes()
    assert [value.numpy().tobytes() for value in restored["cuda"]] == [
        value.numpy().tobytes() for value in expected["cuda"]
    ]
