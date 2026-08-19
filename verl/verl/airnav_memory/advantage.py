"""Episode-level temporal credit assignment followed by segment-wise GRPO."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("airnav_episode_grpo")
def compute_airnav_episode_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Optional[np.ndarray] = None,
    episode_uid: Optional[np.ndarray] = None,
    rollout_id: Optional[np.ndarray] = None,
    segment_index: Optional[np.ndarray] = None,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discount within rollout, then normalize across rollouts at the same segment index."""
    del index
    if config is None:
        raise ValueError("airnav_episode_grpo requires algorithm config")
    if episode_uid is None or rollout_id is None or segment_index is None:
        raise ValueError("airnav_episode_grpo requires episode_uid, rollout_id and segment_index")
    beta = float(config.get("airnav_return_beta", 0.95))
    epsilon = float(config.get("airnav_adv_epsilon", 1e-6))
    immediate_rewards = token_level_rewards.sum(dim=-1)

    trajectory_rows: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in range(len(immediate_rewards)):
        trajectory_rows[(str(episode_uid[row]), int(rollout_id[row]))].append(row)

    discounted = torch.zeros_like(immediate_rewards)
    for rows in trajectory_rows.values():
        rows.sort(key=lambda row: int(segment_index[row]), reverse=True)
        running = torch.zeros((), device=immediate_rewards.device, dtype=immediate_rewards.dtype)
        for row in rows:
            running = immediate_rewards[row] + beta * running
            discounted[row] = running

    group_rows: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in range(len(immediate_rewards)):
        group_rows[(str(episode_uid[row]), int(segment_index[row]))].append(row)

    normalized = torch.zeros_like(discounted)
    for rows in group_rows.values():
        values = discounted[rows]
        if len(rows) <= 1:
            continue
        mean = values.mean()
        # Population standard deviation exactly matches the formula in the experiment definition.
        std = torch.sqrt(torch.mean((values - mean) ** 2))
        normalized[rows] = (values - mean) / (std + epsilon)

    advantages = normalized.unsqueeze(-1) * response_mask
    return advantages, advantages
