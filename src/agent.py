import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model import build_q_network


class DQNAgent:
    def __init__(self, obs_shape, n_actions, lr=1e-4, batch_size=32, buffer_size=50000, min_replay_size=1000, target_update_freq=1000):
        self.obs_shape = obs_shape
        self.n_actions = int(n_actions)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.gamma = 0.99

        self.online_net = build_q_network(obs_shape, self.n_actions)
        self.target_net = build_q_network(obs_shape, self.n_actions)
        self.target_net.set_weights(self.online_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = keras.losses.Huber()
        self.replay_buffer = []
        self.train_steps = 0

    def act(self, state, eps=0.0):
        if np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))
        q_values = self.online_net(np.expand_dims(state, axis=0), training=False)[0]
        return int(np.argmax(q_values.numpy()))

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def can_train(self):
        return len(self.replay_buffer) >= max(self.min_replay_size, self.batch_size)

    def train_step(self):
        if not self.can_train():
            return None

        batch = np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])

        states = tf.convert_to_tensor(np.array(states, dtype=np.float32))
        actions = tf.convert_to_tensor(np.array(actions, dtype=np.int32))
        rewards = tf.convert_to_tensor(np.array(rewards, dtype=np.float32))
        next_states = tf.convert_to_tensor(np.array(next_states, dtype=np.float32))
        dones = tf.convert_to_tensor(np.array(dones, dtype=np.float32))

        with tf.GradientTape() as tape:
            q_values = self.online_net(states, training=True)
            action_q = tf.reduce_sum(q_values * tf.one_hot(actions, self.n_actions), axis=1)

            next_q_values = self.target_net(next_states, training=False)
            next_actions = tf.argmax(self.online_net(next_states, training=False), axis=1)
            next_q = tf.reduce_sum(tf.one_hot(next_actions, self.n_actions) * next_q_values, axis=1)
            targets = rewards + (1.0 - dones) * self.gamma * next_q

            loss = self.loss_fn(targets, action_q)

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.set_weights(self.online_net.get_weights())

        return float(loss.numpy())

    def fit(self, env, nb_steps, checkpoint_dir=None, log_dir=None):
        os.makedirs(checkpoint_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'), exist_ok=True)
        os.makedirs(log_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), exist_ok=True)

        obs, _ = env.reset()
        for step in range(nb_steps):
            action = self.act(obs, eps=max(0.1, 1.0 - step / max(nb_steps, 1)))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.store(obs, action, reward, next_obs, done)
            if self.can_train():
                self.train_step()
            obs = next_obs
            if done:
                obs, _ = env.reset()

        self.save_checkpoint(os.path.join(checkpoint_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints'), 'weights.weights.h5'))

    def test(self, env, nb_episodes=1, visualize=False):
        rewards = []
        for _ in range(nb_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = self.act(obs, eps=0.0)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        print(f'Average reward over {nb_episodes} episode(s): {np.mean(rewards):.2f}')

    def load_weights(self, path):
        if path.endswith('.weights.h5'):
            src = path
        elif path.endswith('.h5'):
            src = path[:-3] + '.weights.h5'
        else:
            src = path + '.weights.h5'
        self.online_net.load_weights(src)
        self.target_net.set_weights(self.online_net.get_weights())

    def save_weights(self, path):
        if path.endswith('.weights.h5'):
            out = path
        elif path.endswith('.h5'):
            out = path[:-3] + '.weights.h5'
        else:
            out = path + '.weights.h5'
        self.online_net.save_weights(out)

    def load_checkpoint(self, path):
        if os.path.exists(path):
            self.load_weights(path)
            return path
        return None

    def setup_checkpoint(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def save_checkpoint(self, path=None):
        if path is None:
            if not hasattr(self, 'checkpoint_dir'):
                raise RuntimeError('Checkpoint manager not configured. Call setup_checkpoint() first.')
            path = os.path.join(self.checkpoint_dir, 'weights.weights.h5')
        self.save_weights(path)
        return path
