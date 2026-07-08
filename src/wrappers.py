import collections
from typing import Tuple

import gymnasium as gym
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    from PIL import Image
    _HAS_CV2 = False


class PreprocessFrame(gym.ObservationWrapper):
    """Convert Atari RGB frame to grayscale, resize to 84x84 and normalize to [0,1]."""
    def __init__(self, env, shape: Tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.shape = shape
        obs_shape = (shape[0], shape[1], 1)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

    def observation(self, obs):
        img = obs.astype(np.uint8)
        if _HAS_CV2:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        else:
            pil = Image.fromarray(img)
            pil = pil.convert('L')
            pil = pil.resize(self.shape[::-1], Image.BILINEAR)
            resized = np.array(pil)

        resized = resized.astype(np.float32) / 255.0
        return np.expand_dims(resized, axis=-1)


class FrameStack(gym.Wrapper):
    """Stack k last frames along the channel dimension (channels-last)."""
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shp[0], shp[1], shp[2] * k), dtype=np.float32)
        self.frames = collections.deque(maxlen=k)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)
