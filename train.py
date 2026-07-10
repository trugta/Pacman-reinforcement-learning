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


def train_loop(env_id='ALE/MsPacman-v5', num_steps=10000, num_episodes=None, eval_every=None, 
               checkpoint_dir=None, log_dir=None, resume_checkpoint=None, eps_decay=250000, **agent_kwargs):
    """Train DQN agent on Atari environment.
    
    Args:
        env_id: Gymnasium environment ID
        num_steps: Number of environment steps to train (if num_episodes is None)
        num_episodes: Number of full episodes to train (overrides num_steps if set)
        eval_every: Evaluate every N episodes (only used with num_episodes)
        checkpoint_dir: Directory to save/load checkpoints
        log_dir: Directory for logs
        resume_checkpoint: Path to checkpoint to resume training from
        eps_decay: Number of steps for epsilon decay schedule
        **agent_kwargs: Additional keyword arguments for DQNAgent
    """
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    log_dir = log_dir or LOGS_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = make_env(env_id, render_mode=None)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    lr = agent_kwargs.get('lr', 1e-4)
    if lr > 1e-4:
        print(
            f'Warning: lr={lr} is higher than typical for Adam with DQN '
            f'(recommended: 1e-4 to 6.25e-5). The original DeepMind rate (2.5e-4) '
            f'was tuned for RMSProp, not Adam. Consider lr=1e-4 for more stable training.'
        )
    agent = DQNAgent(obs_shape, n_actions, eps_decay=eps_decay, **agent_kwargs)
    agent.setup_checkpoint(checkpoint_dir)

    start_step = 0
    
    # Load checkpoint if resuming
    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            result = agent.load_checkpoint(resume_checkpoint)
            if result:
                start_step = agent.env_steps
                print(f'Resuming from checkpoint at step {start_step}')
        else:
            print(f'Warning: checkpoint {resume_checkpoint} not found, starting from scratch')
    
    if num_episodes is not None:
        # True episode-based training
        print(f'Starting TensorFlow/Keras DQN training for {num_episodes} episodes (eps_decay={eps_decay} steps)...')
        agent.fit(env, nb_episodes=num_episodes, checkpoint_dir=checkpoint_dir, log_dir=log_dir, 
                  start_step=start_step, eval_every=eval_every)
    else:
        # Step-based training
        print(f'Starting TensorFlow/Keras DQN training for {num_steps} steps (eps_decay={eps_decay} steps)...')
        agent.fit(env, nb_steps=num_steps, checkpoint_dir=checkpoint_dir, log_dir=log_dir, start_step=start_step)
    
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TensorFlow/Keras DQN agent on ALE/MsPacman-v5')
    
    # Training configuration
    parser.add_argument('--steps', type=int, default=10000, help='number of training steps (default: 10000)')
    parser.add_argument('--episodes', type=int, default=None, help='number of training episodes (overrides --steps if set)')
    parser.add_argument('--eval-every', type=int, default=None, help='evaluate every N episodes (requires --episodes)')
    parser.add_argument('--eps-decay', type=int, default=250000, help='number of steps for epsilon decay (default: 250000)')
    
    # Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--buffer-size', type=int, default=50000, help='replay buffer size (default: 50000)')
    parser.add_argument('--min-replay-size', type=int, default=1000, help='minimum replay buffer size before training (default: 1000)')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='directory to save/load checkpoints')
    parser.add_argument('--resume-checkpoint', type=str, default=None, help='path to checkpoint to resume training from')
    parser.add_argument('--logs-dir', type=str, default=None, help='directory for training logs')
    
    args = parser.parse_args()

    train_loop(
        num_steps=args.steps,
        num_episodes=args.episodes,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.logs_dir,
        resume_checkpoint=args.resume_checkpoint,
        eps_decay=args.eps_decay,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_replay_size=args.min_replay_size,
    )
