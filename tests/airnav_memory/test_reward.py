import math

from gsamllavanav.space import Pose4D
from verl.airnav_memory.reward import compute_segment_reward, compute_terminal_reward


class DummyRaster:
    def index(self, x, y):
        return int(y), int(x)


def test_original_airnav_reward_components():
    pose = Pose4D(0, 0, 50, 0)
    reward = compute_segment_reward(
        pose,
        ["MOVE_FORWARD"],
        ["MOVE_FORWARD"],
        DummyRaster(),
        format_valid=True,
    )
    assert math.isclose(reward.distance, 1.0, rel_tol=1e-5)
    assert reward.yaw == 1.0
    assert reward.format == 0.0
    assert math.isclose(reward.total, 2.0, rel_tol=1e-5)


def test_invalid_format_gets_zero_reward():
    reward = compute_segment_reward(Pose4D(0, 0, 50, 0), [], ["STOP"], DummyRaster(), False)
    assert reward.total == 0.0


def test_segment_penalty_and_terminal_reward():
    reward = compute_segment_reward(
        Pose4D(0, 0, 50, 0),
        ["MOVE_FORWARD"],
        ["MOVE_FORWARD"],
        DummyRaster(),
        True,
        segment_penalty=0.05,
    )
    assert math.isclose(reward.total, 1.95, rel_tol=1e-5)
    assert compute_terminal_reward(terminal=False, success=False) == 0.0
    assert compute_terminal_reward(terminal=True, success=True) == 5.0
    assert compute_terminal_reward(terminal=True, success=False) == -2.0
