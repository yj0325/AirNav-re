"""Parsing and validation for the joint memory/navigation language action."""

from __future__ import annotations

import json
from dataclasses import dataclass


NAVIGATION_ACTIONS = frozenset({"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"})
MEMORY_ACTIONS = frozenset(
    {"APPEND_CURRENT", "DROP_1", "DROP_2", "DROP_3", "DROP_4", "DROP_CURRENT"}
)


@dataclass(frozen=True)
class JointAction:
    memory_action: str
    navigation_actions: tuple[str, ...]


def _extract_json_object(text: str) -> dict:
    """Extract the first balanced JSON object, ignoring surrounding model chatter."""
    start = text.find("{")
    if start < 0:
        raise ValueError("missing JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("unterminated JSON object")


def parse_joint_action(text: str, memory_size: int, capacity: int = 4) -> JointAction:
    """Parse the model output and enforce the state-dependent memory action set."""
    payload = _extract_json_object(text.replace("'", '"'))
    if set(payload) != {"memory_action", "navigation_actions"}:
        raise ValueError("joint action must contain exactly memory_action and navigation_actions")

    memory_action = payload["memory_action"]
    actions = payload["navigation_actions"]
    if not isinstance(memory_action, str) or memory_action not in MEMORY_ACTIONS:
        raise ValueError(f"invalid memory action: {memory_action!r}")
    if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
        raise ValueError("navigation_actions must contain 1 to 8 actions")
    if not all(isinstance(action, str) and action in NAVIGATION_ACTIONS for action in actions):
        raise ValueError("invalid navigation action")
    if "STOP" in actions[:-1]:
        raise ValueError("STOP may only be the final navigation action")

    if memory_size < capacity and memory_action != "APPEND_CURRENT":
        raise ValueError("APPEND_CURRENT is required while the memory window is not full")
    if memory_size >= capacity and memory_action == "APPEND_CURRENT":
        raise ValueError("APPEND_CURRENT is invalid when the memory window is full")

    return JointAction(memory_action=memory_action, navigation_actions=tuple(actions))
