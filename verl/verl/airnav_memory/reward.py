"""AirNav navigation shaping and episode-terminal rewards."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gsamllavanav.actions import DiscreteAction
from gsamllavanav.space import Pose4D
from gsamllavanav.teacher.algorithm.lookahead import LookaheadTeacherParams, lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose


ACTION_TO_ENV = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
}


@dataclass(frozen=True)
class RewardBreakdown:
    distance: float
    yaw: float
    format: float
    segment: float = 0.0
    terminal: float = 0.0

    @property
    def total(self) -> float:
        return self.distance + self.yaw + self.format + self.segment + self.terminal


def _dump_yaw(yaw: float) -> float:
    return (yaw + math.pi) % (2 * math.pi) - math.pi


def apply_named_actions(pose: Pose4D, actions: list[str] | tuple[str, ...]) -> Pose4D:
    current = Pose4D(*pose)
    for action in actions:
        if action == "MOVE_FORWARD":
            current = Pose4D(
                current.x + 5 * math.cos(current.yaw),
                current.y + 5 * math.sin(current.yaw),
                current.z,
                current.yaw,
            )
        elif action == "TURN_LEFT":
            current = Pose4D(current.x, current.y, current.z, _dump_yaw(current.yaw + math.pi / 6))
        elif action == "TURN_RIGHT":
            current = Pose4D(current.x, current.y, current.z, _dump_yaw(current.yaw - math.pi / 6))
        elif action == "STOP":
            break
    return current


def dynamic_teacher_actions(
    current_pose: Pose4D,
    teacher_trajectory: list[Pose4D],
    max_actions: int = 8,
) -> list[str]:
    """Generate the original lookahead teacher's next local block from an online state."""
    points = np.asarray([[pose.x, pose.y, current_pose.z] for pose in teacher_trajectory], dtype=np.float64)
    distances = np.linalg.norm(points[:, :2] - np.asarray(current_pose.xy), axis=-1)
    points = points[int(np.argmin(distances)) :]
    pose = Pose4D(*current_pose)
    params = LookaheadTeacherParams(lookahead=1)
    actions: list[str] = []
    for _ in range(max_actions):
        action = lookahead_discrete_action(pose, points, params)
        if action in {DiscreteAction.GO_UP, DiscreteAction.GO_DOWN}:
            # AirNav's action space is planar; fixed-altitude data should never enter this branch.
            action = DiscreteAction.MOVE_FORWARD
        actions.append(action.name)
        if action is DiscreteAction.STOP:
            break
        pose = _moved_pose(pose, *action.value)
        nearest = int(np.argmin(np.linalg.norm(points[:, :2] - np.asarray(pose.xy), axis=-1)))
        points = points[nearest:]
    return actions


def compute_segment_reward(
    current_pose: Pose4D,
    predicted_actions: list[str] | tuple[str, ...],
    teacher_actions: list[str],
    raster,
    format_valid: bool,
    segment_penalty: float = 0.0,
) -> RewardBreakdown:
    if not format_valid:
        return RewardBreakdown(distance=0.0, yaw=0.0, format=0.0)

    gt_pose = apply_named_actions(current_pose, teacher_actions)
    pred_pose = apply_named_actions(current_pose, predicted_actions)
    cur_row, cur_col = raster.index(current_pose.x, current_pose.y)
    gt_row, gt_col = raster.index(gt_pose.x, gt_pose.y)
    pred_row, pred_col = raster.index(pred_pose.x, pred_pose.y)

    original_distance = math.hypot(gt_col - cur_col, gt_row - cur_row) + 1e-6
    new_distance = math.hypot(gt_col - pred_col, gt_row - pred_row)
    distance_reward = max((original_distance - new_distance) / original_distance, 0.0)

    yaw_delta_deg = math.degrees(_dump_yaw(gt_pose.yaw - pred_pose.yaw))
    yaw_reward = max(1.0 - abs(yaw_delta_deg) / 60.0, 0.0)

    return RewardBreakdown(
        distance=distance_reward,
        yaw=yaw_reward,
        format=0.0,
        segment=-abs(float(segment_penalty)),
    )


def compute_terminal_reward(
    *,
    terminal: bool,
    success: bool,
    success_reward: float = 5.0,
    failure_reward: float = -2.0,
) -> float:
    """Return a navigation-level terminal reward, never a memory-label reward."""
    if not terminal:
        return 0.0
    return float(success_reward if success else failure_reward)
