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


def load_agent_artifacts(agent, checkpoint_dir=None):
    """Load the trained agent from checkpoint.
    
    Args:
        agent: DQNAgent instance to load into
        checkpoint_dir: Directory containing checkpoint files
    
    Returns:
        Path to loaded checkpoint
    """
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.h5')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found at {ckpt_path}.\n'
            f'Train an agent first with: python train.py --steps 10000'
        )
    
    result = agent.load_checkpoint(checkpoint_dir)
    if not result:
        raise RuntimeError(f'Failed to load checkpoint from {checkpoint_dir}')
    
    return result


def play(env_id='ALE/MsPacman-v5', episodes=1, checkpoint_dir=None, render=False):
    """Run trained agent for evaluation.
    
    Args:
        env_id: Gymnasium environment ID
        episodes: Number of episodes to play
        checkpoint_dir: Directory containing checkpoint
        render: Whether to render the environment
    """
    env = make_env(env_id, render_mode='human' if render else None)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(obs_shape, n_actions)
    load_agent_artifacts(agent, checkpoint_dir=checkpoint_dir)
    reward = agent.test(env, nb_episodes=episodes)
    if reward is not None:
        print(f'Average reward over {episodes} episode(s): {reward:.2f}')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a trained MsPacman agent with TensorFlow/Keras.')
    parser.add_argument('--checkpoint-dir', default=None, help='Directory containing checkpoint (default: checkpoints/)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play (default: 1)')
    parser.add_argument('--env-id', default='ALE/MsPacman-v5', help='Gymnasium environment id')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering (headless mode)')
    args = parser.parse_args()

    play(env_id=args.env_id, episodes=args.episodes, checkpoint_dir=args.checkpoint_dir, render=not args.no_render)