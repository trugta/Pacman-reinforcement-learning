import collections
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model import build_q_network


def _json_safe(obj):
    """Recursively convert numpy scalar/array types to native Python types
    so json.dump() doesn't choke on things like np.float32 that can appear
    inside keras.optimizers.serialize()'s output (observed with TF 2.10's
    legacy optimizer implementation).
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):  # covers np.float32, np.int64, etc.
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class DQNAgent:
    def __init__(self, obs_shape, n_actions, lr=1e-4, batch_size=32, buffer_size=50000, min_replay_size=1000, target_update_freq=1000, eps_decay=250000):
        self.obs_shape = obs_shape
        self.n_actions = int(n_actions)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.gamma = 0.99
        self.eps_decay = eps_decay

        self.online_net = build_q_network(obs_shape, self.n_actions)
        self.target_net = build_q_network(obs_shape, self.n_actions)
        self.target_net.set_weights(self.online_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = keras.losses.Huber()
        self.online_net.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.replay_buffer = collections.deque(maxlen=buffer_size)
        self.train_steps = 0  # Counts gradient updates (for target-network sync)
        self.env_steps = 0    # Counts environment interactions (for epsilon decay, resume)

    def act(self, state, eps=0.0):
        if np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))
        q_values = self.online_net(np.expand_dims(state, axis=0), training=False)[0]
        return int(np.argmax(q_values.numpy()))

    def store(self, state, action, reward, next_state, terminated):
        """Store transition in replay buffer. Store terminated (not truncated) for proper Bellman masking."""
        self.replay_buffer.append((state, action, reward, next_state, terminated))

    def can_train(self):
        return len(self.replay_buffer) >= max(self.min_replay_size, self.batch_size)

    def train_step(self):
        if not self.can_train():
            return None

        batch = np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
        states, actions, rewards, next_states, terminated = zip(*[self.replay_buffer[i] for i in batch])

        states = tf.convert_to_tensor(np.array(states, dtype=np.float32))
        actions = tf.convert_to_tensor(np.array(actions, dtype=np.int32))
        rewards = tf.convert_to_tensor(np.array(rewards, dtype=np.float32))
        next_states = tf.convert_to_tensor(np.array(next_states, dtype=np.float32))
        terminated = tf.convert_to_tensor(np.array(terminated, dtype=np.float32))

        with tf.GradientTape() as tape:
            q_values = self.online_net(states, training=True)
            action_q = tf.reduce_sum(q_values * tf.one_hot(actions, self.n_actions), axis=1)

            next_q_values = self.target_net(next_states, training=False)
            next_actions = tf.argmax(self.online_net(next_states, training=False), axis=1)
            next_q = tf.reduce_sum(tf.one_hot(next_actions, self.n_actions) * next_q_values, axis=1)
            targets = rewards + (1.0 - terminated) * self.gamma * next_q

            loss = self.loss_fn(targets, action_q)

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.set_weights(self.online_net.get_weights())

        return float(loss.numpy())

    def _get_epsilon(self, step):
        """Compute epsilon for epsilon-greedy exploration using decay schedule."""
        return max(0.01, 1.0 - step / self.eps_decay)

    def fit(self, env, nb_steps=None, nb_episodes=None, checkpoint_dir=None, log_dir=None, start_step=0, eval_env=None, eval_every=None, log_every=None, checkpoint_every=None):
        """Train agent for either nb_steps environment interactions or nb_episodes full episodes.
        
        Args:
            env: Gymnasium training environment
            nb_steps: Number of environment steps to take (if nb_episodes is None)
            nb_episodes: Number of full episodes to train (overrides nb_steps if set)
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for logs
            start_step: Step to resume from (for checkpointing)
            eval_env: Separate environment for evaluation (if None, uses training env)
            eval_every: Evaluate every N episodes (only with nb_episodes; if None, no eval)
            log_every: Print progress every N env steps (only with nb_steps; if None, no periodic logging)
            checkpoint_every: Save a checkpoint every N env steps (only with nb_steps; if None, only saves at the end)
        """
        os.makedirs(checkpoint_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'), exist_ok=True)
        os.makedirs(log_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)

        self.env_steps = start_step  # Resume from checkpoint's env step count
        obs, info = env.reset()
        current_lives = info.get('lives', None)
        
        if nb_episodes is not None:
            # Episode-based training
            episode_count = 0
            while episode_count < nb_episodes:
                episode_reward = 0.0
                obs, info = env.reset()
                current_lives = info.get('lives', None)
                while True:
                    eps = self._get_epsilon(self.env_steps)
                    action = self.act(obs, eps=eps)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    next_lives = info.get('lives', None)
                    life_lost = (next_lives is not None and current_lives is not None
                                 and next_lives < current_lives)
                    clipped_reward = float(np.clip(reward, -1.0, 1.0))
                    self.store(obs, action, clipped_reward, next_obs, terminated or life_lost)
                    if self.can_train():
                        self.train_step()
                    self.env_steps += 1
                    current_lives = next_lives
                    obs = next_obs
                    if terminated or truncated:
                        break
                
                episode_count += 1
                if episode_count % max(1, nb_episodes // 10) == 0:
                    print(f'Episode {episode_count}/{nb_episodes}, reward: {episode_reward:.2f}')
                
                # Periodic evaluation
                if eval_every is not None and episode_count % eval_every == 0:
                    eval_env_to_use = eval_env or env
                    eval_reward = self._evaluate(eval_env_to_use, nb_episodes=1)
                    print(f'  [Eval at episode {episode_count}] reward: {eval_reward:.2f}')
        else:
            # Step-based training
            checkpoint_path = os.path.join(
                checkpoint_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'),
                'checkpoint.h5',
            )
            episode_reward = 0.0
            recent_episode_rewards = collections.deque(maxlen=20)
            recent_losses = collections.deque(maxlen=100)
            episode_count = 0

            for step in range(start_step, start_step + (nb_steps or 10000)):
                eps = self._get_epsilon(self.env_steps)
                action = self.act(obs, eps=eps)
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                next_lives = info.get('lives', None)
                life_lost = (next_lives is not None and current_lives is not None
                             and next_lives < current_lives)
                clipped_reward = float(np.clip(reward, -1.0, 1.0))
                self.store(obs, action, clipped_reward, next_obs, terminated or life_lost)
                if self.can_train():
                    loss = self.train_step()
                    if loss is not None:
                        recent_losses.append(loss)
                self.env_steps += 1
                current_lives = next_lives
                obs = next_obs

                if terminated or truncated:
                    episode_count += 1
                    recent_episode_rewards.append(episode_reward)
                    episode_reward = 0.0
                    obs, info = env.reset()
                    current_lives = info.get('lives', None)

                if log_every is not None and self.env_steps % log_every == 0:
                    avg_reward = float(np.mean(recent_episode_rewards)) if recent_episode_rewards else float('nan')
                    avg_loss = float(np.mean(recent_losses)) if recent_losses else float('nan')
                    print(
                        f'Step {self.env_steps}/{start_step + (nb_steps or 10000)} '
                        f'(episodes={episode_count}, eps={eps:.3f}, '
                        f'avg_reward(last {len(recent_episode_rewards)} ep)={avg_reward:.2f}, '
                        f'avg_loss(last {len(recent_losses)})={avg_loss:.4f})'
                    )

                if checkpoint_every is not None and self.env_steps % checkpoint_every == 0:
                    self.save_checkpoint(checkpoint_path)

        if self.env_steps < self.eps_decay:
            print(
                f'Warning: eps_decay ({self.eps_decay:,}) exceeds total env_steps trained '
                f'({self.env_steps:,}). Epsilon only reached '
                f'{self._get_epsilon(self.env_steps):.3f} (floor is 0.01). '
                f'Reduce --eps-decay or train longer for full exploration decay.'
            )
        self.save_checkpoint(os.path.join(checkpoint_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'), 'checkpoint.h5'))
    
    def _evaluate(self, env, nb_episodes=1):
        """Run evaluation episodes and return average reward."""
        rewards = []
        for _ in range(nb_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            while True:
                action = self.act(obs, eps=0.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)
        return float(np.mean(rewards))

    def test(self, env, nb_episodes=1):
        """Test agent for nb_episodes."""
        return self._evaluate(env, nb_episodes=nb_episodes)

    def save_checkpoint(self, path=None):
        """Save full checkpoint including optimizer state and train_steps."""
        if path is None:
            if not hasattr(self, 'checkpoint_dir'):
                raise RuntimeError('Checkpoint manager not configured. Call setup_checkpoint() first.')
            path = os.path.join(self.checkpoint_dir, 'checkpoint.h5')
        
        # Handle case where path is a bare filename (no directory component)
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        self.online_net.save(path)
        
        metadata = {
            'train_steps': int(self.train_steps),
            'env_steps': int(self.env_steps),
            'optimizer_config': keras.optimizers.serialize(self.optimizer),
        }
        metadata = _json_safe(metadata)
        metadata_path = path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f'Checkpoint saved: {path} (train_steps={self.train_steps}, env_steps={self.env_steps})')
        return path

    def load_checkpoint(self, path):
        """Load full checkpoint including optimizer state and train_steps."""
        if path is None or not os.path.exists(path):
            return None

        if os.path.isdir(path):
            ckpt_path = os.path.join(path, 'checkpoint.h5')
            if not os.path.exists(ckpt_path):
                return None
        else:
            ckpt_path = path
        
        try:
            self.online_net = keras.models.load_model(ckpt_path)
            self.target_net.set_weights(self.online_net.get_weights())
            
            # CRITICAL: Reassign self.optimizer to the loaded model's optimizer
            # so train_step() uses the restored Adam moments, not a fresh one
            self.optimizer = self.online_net.optimizer
            self.loss_fn = self.online_net.loss
            
            metadata_path = ckpt_path.replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.train_steps = metadata.get('train_steps', 0)
                self.env_steps = metadata.get('env_steps', 0)
            
            print(f'Checkpoint loaded: {ckpt_path} (train_steps={self.train_steps}, env_steps={self.env_steps})')
            return ckpt_path
        except Exception as e:
            print(f'Failed to load checkpoint: {e}')
            return None

    def setup_checkpoint(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir