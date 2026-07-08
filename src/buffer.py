from typing import Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_shape: Tuple[int, int, int], size: int = 100000, batch_size: int = 32):
        self.obs_shape = obs_shape
        self.max_size = size
        self.batch_size = batch_size

        self.states = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((size,), dtype=np.int32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.dones = np.zeros((size,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state.astype(np.uint8, copy=False)
        self.next_states[self.ptr] = next_state.astype(np.uint8, copy=False)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        sample_size = min(self.batch_size, self.size)
        replace = sample_size < self.batch_size
        idxs = np.random.choice(self.size, size=sample_size, replace=replace)
        return (
            self.states[idxs].astype(np.float32, copy=False),
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs].astype(np.float32, copy=False),
            self.dones[idxs],
        )
