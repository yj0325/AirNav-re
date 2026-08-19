from verl.airnav_memory.memory import MemoryWindow


def test_memory_fill_replace_and_drop_current():
    memory = MemoryWindow(capacity=4)
    for frame in range(4):
        memory.update("APPEND_CURRENT", frame)
    memory.update("DROP_2", 4)
    assert memory.frames == [0, 2, 3, 4]
    memory.update("DROP_CURRENT", 5)
    assert memory.frames == [0, 2, 3, 4]
