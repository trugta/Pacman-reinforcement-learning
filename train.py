import argparse
import os

import ale_py
import gymnasium as gym

from src.agent import DQNAgent
from src.wrappers import FrameStack, PreprocessFrame

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

for directory in (CHECKPOINT_DIR, LOGS_DIR):
    os.makedirs(directory, exist_ok=True)


def make_env(env_id='ALE/MsPacman-v5', render_mode=None):
    try:
        gym.register_envs(ale_py)
    except Exception:
        pass
    env = gym.make(env_id, render_mode=render_mode)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)
    return env


def train_loop(env_id='ALE/MsPacman-v5', num_steps=10000, checkpoint_dir=None, log_dir=None, **agent_kwargs):
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    log_dir = log_dir or LOGS_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(env_id, render_mode=None)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(obs_shape, n_actions, **agent_kwargs)
    agent.setup_checkpoint(checkpoint_dir)

    print(f'Starting TensorFlow/Keras DQN training for {num_steps} steps...')
    agent.fit(env, nb_steps=num_steps, checkpoint_dir=checkpoint_dir, log_dir=log_dir)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TensorFlow/Keras DQN agent on ALE/MsPacman-v5')
    parser.add_argument('--steps', type=int, default=10000, help='number of training steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--buffer-size', type=int, default=50000, help='replay buffer size')
    parser.add_argument('--min-replay-size', type=int, default=1000, help='minimum replay buffer size before training')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='directory to save weights checkpoints')
    parser.add_argument('--logs-dir', type=str, default=None, help='directory for training logs')
    args = parser.parse_args()

    train_loop(
        num_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.logs_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_replay_size=args.min_replay_size,
    )
