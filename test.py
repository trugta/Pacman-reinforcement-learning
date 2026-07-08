import argparse
import os
import time

import ale_py
import gymnasium as gym

from src.agent import DQNAgent
from src.wrappers import FrameStack, PreprocessFrame

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')


def make_env(env_id='ALE/MsPacman-v5', render_mode='human'):
    try:
        gym.register_envs(ale_py)
    except Exception:
        pass

    env = gym.make(env_id, render_mode=render_mode)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)
    return env


def load_agent_artifacts(agent, weights_path=None, checkpoint_dir=None):
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    if checkpoint_dir:
        restored = agent.load_checkpoint(checkpoint_dir)
        if restored:
            print(f'Loaded checkpoint: {restored}')
            return restored
        print(f'No checkpoint found in {checkpoint_dir}; trying weights...')

    if weights_path:
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(CHECKPOINT_DIR, weights_path)
        agent.load_weights(weights_path)
        print(f'Loaded weights: {weights_path}')
        return weights_path

    raise FileNotFoundError('No checkpoint or weights file could be loaded.')


def play(weights_path='best_mspacman', env_id='ALE/MsPacman-v5', episodes=1, checkpoint_dir=None, render=True):
    env = make_env(env_id, render_mode='human' if render else None)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(obs_shape, n_actions)
    load_agent_artifacts(agent, weights_path=weights_path, checkpoint_dir=checkpoint_dir)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(obs, eps=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            if render:
                time.sleep(0.01)
        print(f'Episode {ep + 1} reward: {ep_reward}')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a trained MsPacman agent with the same wrappers as training.')
    parser.add_argument('--weights', default='best_mspacman', help='Base weights path for the trained Q-network')
    parser.add_argument('--checkpoint-dir', default=None, help='Directory containing full training checkpoints')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play')
    parser.add_argument('--env-id', default='ALE/MsPacman-v5', help='Gymnasium environment id')
    parser.add_argument('--no-render', action='store_true', help='Run without rendering the environment')
    args = parser.parse_args()

    play(
        weights_path=args.weights,
        env_id=args.env_id,
        episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
        render=not args.no_render,
    )
