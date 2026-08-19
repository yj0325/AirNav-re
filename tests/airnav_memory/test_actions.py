import json

import pytest

from verl.airnav_memory.actions import parse_joint_action


def test_joint_action_before_window_is_full():
    action = parse_joint_action(
        json.dumps(
            {
                "memory_action": "APPEND_CURRENT",
                "navigation_actions": ["TURN_LEFT", "MOVE_FORWARD"],
            }
        ),
        memory_size=2,
    )
    assert action.memory_action == "APPEND_CURRENT"
    assert action.navigation_actions[-1] == "MOVE_FORWARD"


def test_joint_action_full_window_replacement():
    action = parse_joint_action(
        '{"memory_action":"DROP_2","navigation_actions":["MOVE_FORWARD","STOP"]}',
        memory_size=4,
    )
    assert action.memory_action == "DROP_2"


@pytest.mark.parametrize(
    "response,memory_size",
    [
        ('{"memory_action":"DROP_CURRENT","navigation_actions":["MOVE_FORWARD"]}', 1),
        ('{"memory_action":"APPEND_CURRENT","navigation_actions":["MOVE_FORWARD"]}', 4),
        ('{"memory_action":"DROP_1","navigation_actions":["STOP","MOVE_FORWARD"]}', 4),
    ],
)
def test_joint_action_rejects_invalid_state_or_stop(response, memory_size):
    with pytest.raises(ValueError):
        parse_joint_action(response, memory_size=memory_size)
