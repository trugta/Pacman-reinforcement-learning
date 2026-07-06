from typing import Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_shape: Tuple[int, int, int], size: int = 100000, batch_size: int = 32):
        self.obs_shape = obs_shape
        self.max_size = size
        self.batch_size = batch_size

        self.states = np.zeros((size, *obs_shape), dtype=np.float32)
        self.next_states = np.zeros((size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((size,), dtype=np.int32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.dones = np.zeros((size,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )
