"""Fixed-capacity visual memory state transition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryWindow:
    capacity: int = 4
    frames: list[Any] = field(default_factory=list)

    def update(self, action: str, current_frame: Any) -> None:
        if len(self.frames) < self.capacity:
            if action != "APPEND_CURRENT":
                raise ValueError("memory must append the current frame until full")
            self.frames.append(current_frame)
            return

        if action == "DROP_CURRENT":
            return
        if not action.startswith("DROP_"):
            raise ValueError(f"invalid full-window action: {action}")
        slot = int(action.removeprefix("DROP_")) - 1
        if not 0 <= slot < self.capacity:
            raise ValueError(f"invalid memory slot: {slot + 1}")
        del self.frames[slot]
        self.frames.append(current_frame)

    def snapshot(self) -> list[Any]:
        return list(self.frames)
