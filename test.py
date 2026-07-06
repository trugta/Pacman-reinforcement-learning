import time

import gymnasium as gym

from wrappers import PreprocessFrame, FrameStack
from agent import DQNAgent


def play(weights_path: str, env_id='ALE/MsPacman-v5', episodes: int = 1):
    env = gym.make(env_id, render_mode='human')
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(obs_shape, n_actions)
    agent.load_weights(weights_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(obs, eps=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            time.sleep(0.01)
        print(f'Episode {ep+1} reward: {ep_reward}')

    env.close()


if __name__ == '__main__':
    # pass base name; DQNAgent.load_weights will accept and append suffix
    play('best_mspacman', episodes=1)
