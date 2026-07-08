import argparse
import os

import ale_py
import gymnasium as gym

from src.agent import DQNAgent
from src.wrappers import FrameStack, PreprocessFrame

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')


def make_env(env_id='ALE/MsPacman-v5', render_mode=None):
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
    if weights_path:
        if not os.path.isabs(weights_path):
            if os.path.exists(weights_path):
                resolved_path = weights_path
            elif os.path.exists(os.path.join(checkpoint_dir, weights_path)):
                resolved_path = os.path.join(checkpoint_dir, weights_path)
            else:
                resolved_path = os.path.join(checkpoint_dir, weights_path)
            weights_path = resolved_path
        agent.load_weights(weights_path)
        print(f'Loaded weights: {weights_path}')
        return weights_path
    if checkpoint_dir:
        restored = agent.load_checkpoint(checkpoint_dir)
        if restored:
            print(f'Loaded checkpoint: {restored}')
            return restored
    raise FileNotFoundError('No checkpoint or weights file could be loaded.')


def play(weights_path='weights.h5', env_id='ALE/MsPacman-v5', episodes=1, checkpoint_dir=None, render=False):
    env = make_env(env_id, render_mode='human' if render else None)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(obs_shape, n_actions)
    load_agent_artifacts(agent, weights_path=weights_path, checkpoint_dir=checkpoint_dir)
    agent.test(env, nb_episodes=episodes, visualize=render)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a trained MsPacman agent with TensorFlow/Keras.')
    parser.add_argument('--weights', default='weights.h5', help='Path to the saved DQN weights file')
    parser.add_argument('--checkpoint-dir', default=None, help='Directory containing saved weights checkpoints')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play')
    parser.add_argument('--env-id', default='ALE/MsPacman-v5', help='Gymnasium environment id')
    parser.add_argument('--render', action='store_true', help='Render the environment while evaluating')
    args = parser.parse_args()

    play(weights_path=args.weights, env_id=args.env_id, episodes=args.episodes, checkpoint_dir=args.checkpoint_dir, render=args.render)
