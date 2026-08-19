from types import SimpleNamespace

import numpy as np
import torch

from verl.airnav_memory.advantage import compute_airnav_episode_grpo_advantage


def test_discount_then_segment_group_normalization():
    # rollout 0 rewards [1, 2], rollout 1 rewards [3, 0], beta=.5
    rewards = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 0.0]])
    mask = torch.ones_like(rewards)
    config = SimpleNamespace(get=lambda key, default=None: {"airnav_return_beta": 0.5}.get(key, default))
    advantages, _ = compute_airnav_episode_grpo_advantage(
        rewards,
        mask,
        episode_uid=np.array(["e", "e", "e", "e"], dtype=object),
        rollout_id=np.array([0, 0, 1, 1], dtype=object),
        segment_index=np.array([0, 1, 0, 1], dtype=object),
        config=config,
    )
    # Returns are [2, 2] vs [3, 0], population-normalized to [-1,+1] and [+1,-1].
    assert torch.allclose(advantages[:, 0], torch.tensor([-1.0, 1.0, 1.0, -1.0]), atol=1e-5)
