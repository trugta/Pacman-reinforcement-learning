from rl.memory import SequentialMemory


def build_memory(limit: int = 50000, window_length: int = 1):
    return SequentialMemory(limit=limit, window_length=window_length)
