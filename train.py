import argparse
import os
import random
import time

import ale_py
import gymnasium as gym
import numpy as np
import tensorflow as tf

from src.agent import DQNAgent
from src.wrappers import FrameStack, PreprocessFrame

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

for directory in (CHECKPOINT_DIR, LOGS_DIR, SRC_DIR):
    os.makedirs(directory, exist_ok=True)


def train_loop(env_id='ALE/MsPacman-v5', num_episodes=1000, eval_every=50, eps_decay=250000, checkpoint_dir=None, resume=False, **agent_kwargs):
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    # register ALE envs so 'ALE/...' namespaces are available
    try:
        gym.register_envs(ale_py)
    except Exception:
        # if already registered, ignore
        pass

    env = gym.make(env_id, render_mode=None)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # pick device for training (prefer GPU if available)
    try:
        gpus = tf.config.list_physical_devices('GPU')
    except Exception:
        gpus = []
    device = '/GPU:0' if gpus else '/CPU:0'

    # create agent on the selected device
    with tf.device(device):
        agent = DQNAgent(obs_shape, n_actions, **agent_kwargs)

    # setup checkpointing if requested
    if checkpoint_dir:
        agent.setup_checkpoint(checkpoint_dir)
        if resume:
            restored = agent.load_checkpoint(checkpoint_dir)
            if restored:
                print(f'Restored checkpoint: {restored}')
            else:
                print('No checkpoint found to restore; starting fresh.')

    # Warmup random experience
    print('Collecting initial random experience...')
    obs, _ = env.reset()
    while agent.replay.size < agent.min_replay_size:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs, _ = env.reset()

    print('Starting training...')
    eps_initial = 1.0
    eps_final = 0.01
    total_steps = 0
    eps = eps_initial

    rewards_history = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, eps=0.0)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            total_steps += 1

            # linear epsilon annealing between eps_initial -> eps_final
            frac = min(total_steps / eps_decay, 1.0)
            eps = eps_initial + frac * (eps_final - eps_initial)

            if agent.can_train():
                # ensure training step runs on chosen device
                with tf.device(device):
                    loss = agent.train_step()

        rewards_history.append(ep_reward)

        if ep % 10 == 0:
            avg = np.mean(rewards_history[-100:]) if rewards_history else 0.0
            print(f'Episode {ep:4d} | Reward {ep_reward:.1f} | Avg100 {avg:.2f} | Eps {eps:.3f}')

        if ep % eval_every == 0:
            avg_eval = evaluate(agent, env_id, episodes=3, render=False)
            print(f'-- Eval after {ep} episodes: {avg_eval:.2f}')
            # save best model
            # For simplicity saving every eval
            if checkpoint_dir:
                ckpt_path = agent.save_checkpoint()
                print(f'Checkpoint saved: {ckpt_path}')
            else:
                # fallback to legacy weights-only saving
                weights_path = os.path.join(CHECKPOINT_DIR, 'best_mspacman')
                agent.save_weights(weights_path)

    env.close()


def evaluate(agent, env_id='ALE/MsPacman-v5', episodes=5, render=False):
    env = gym.make(env_id, render_mode='human' if render else None)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)
    total = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(obs, eps=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total.append(ep_reward)
    env.close()
    return float(np.mean(total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN/DDQN on ALE/MsPacman-v5')
    parser.add_argument('--episodes', type=int, default=200, help='number of training episodes')
    parser.add_argument('--eval-every', type=int, default=25, help='evaluation frequency (episodes)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--eps-decay', type=int, default=250000, help='epsilon decay steps for linear annealing')
    parser.add_argument('--buffer-size', type=int, default=50000, help='replay buffer size')
    parser.add_argument('--min-replay-size', type=int, default=5000, help='minimum replay buffer size before training')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='directory to save full checkpoints (model + optimizer)')
    parser.add_argument('--logs-dir', type=str, default=None, help='directory for TensorBoard or metrics logs')
    parser.add_argument('--resume-checkpoint', action='store_true', help='if set, attempt to restore from latest checkpoint in --checkpoint-dir before training')
    args = parser.parse_args()

    logs_dir = args.logs_dir or LOGS_DIR
    os.makedirs(logs_dir, exist_ok=True)

    agent_kwargs = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'buffer_size': args.buffer_size,
        'min_replay_size': args.min_replay_size,
    }

    train_loop(num_episodes=args.episodes, eval_every=args.eval_every, eps_decay=args.eps_decay, checkpoint_dir=args.checkpoint_dir, resume=args.resume_checkpoint, **agent_kwargs)
