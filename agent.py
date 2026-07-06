import numpy as np
import tensorflow as tf
from tensorflow import keras

from buffer import ReplayBuffer
from model import build_q_network


# GPU memory growth precaution
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


class DQNAgent:
    def __init__(self,
                 obs_shape,
                 n_actions,
                 lr=1e-4,
                 gamma=0.99,
                 batch_size=32,
                 buffer_size=100000,
                 min_replay_size=1000,
                 target_update_freq=1000,
                 double_dqn=True):
        self.obs_shape = obs_shape
        self.n_actions = int(n_actions)
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn

        self.replay = ReplayBuffer(obs_shape, size=buffer_size, batch_size=batch_size)

        # device placement: prefer GPU if available
        try:
            physical_gpus = tf.config.list_physical_devices('GPU')
        except Exception:
            physical_gpus = []
        self.device = '/GPU:0' if physical_gpus else '/CPU:0'

        # Build networks on the selected device
        with tf.device(self.device):
            self.online_net = build_q_network(obs_shape, self.n_actions)
            self.target_net = build_q_network(obs_shape, self.n_actions)
            self.target_net.set_weights(self.online_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.train_steps = 0

    def act(self, state, eps: float = 0.0):
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
        # ensure inference runs on chosen device
        with tf.device(self.device):
            q_vals = self.online_net(state_tensor, training=False).numpy()[0]
        return int(np.argmax(q_vals))

    def store(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)

    def update_target(self):
        self.target_net.set_weights(self.online_net.get_weights())

    def save_weights(self, path: str):
        # Keras (HDF5) weights filenames convention in newer Keras requires
        # ending with '.weights.h5' for HDF5 format. Append suffix if missing.
        if path.endswith('.weights.h5'):
            out = path
        elif path.endswith('.h5'):
            out = path[:-3] + '.weights.h5'
        else:
            out = path + '.weights.h5'
        self.online_net.save_weights(out)

    def load_weights(self, path: str):
        # Accept either the full filename or without the '.weights.h5' suffix.
        if path.endswith('.weights.h5'):
            src = path
        elif path.endswith('.h5'):
            src = path[:-3] + '.weights.h5'
        else:
            src = path + '.weights.h5'
        self.online_net.load_weights(src)
        self.target_net.set_weights(self.online_net.get_weights())

    # ----- Full checkpointing (model + optimizer) -----
    def setup_checkpoint(self, ckpt_dir: str, max_to_keep: int = 5):
        # Prepare a Checkpoint and CheckpointManager for saving/loading full state
        self._ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.online_net)
        self._ckpt_manager = tf.train.CheckpointManager(self._ckpt, ckpt_dir, max_to_keep=max_to_keep)

    def save_checkpoint(self):
        if not hasattr(self, '_ckpt_manager'):
            raise RuntimeError('Checkpoint manager not configured. Call setup_checkpoint() first.')
        path = self._ckpt_manager.save()
        return path

    def load_checkpoint(self, ckpt_dir: str):
        # Restore latest checkpoint from directory (if present). Returns restored path or None.
        tmp_ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.online_net)
        tmp_manager = tf.train.CheckpointManager(tmp_ckpt, ckpt_dir, max_to_keep=5)
        if tmp_manager.latest_checkpoint:
            tmp_ckpt.restore(tmp_manager.latest_checkpoint).expect_partial()
            # sync target network
            self.target_net.set_weights(self.online_net.get_weights())
            # sync train_steps from optimizer iterations if available
            try:
                self.train_steps = int(self.optimizer.iterations.numpy())
            except Exception:
                pass
            return tmp_manager.latest_checkpoint
        return None

    def can_train(self):
        return self.replay.size >= max(self.min_replay_size, self.batch_size)

    def train_step(self):
        s, a, r, s2, d = self.replay.sample()
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        a = tf.convert_to_tensor(a, dtype=tf.int32)
        r = tf.convert_to_tensor(r, dtype=tf.float32)
        s2 = tf.convert_to_tensor(s2, dtype=tf.float32)
        d = tf.convert_to_tensor(d, dtype=tf.float32)

        # Run training on the chosen device to ensure GPU usage when available
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                q_values = self.online_net(s, training=True)
                q_action = tf.reduce_sum(q_values * tf.one_hot(a, self.n_actions), axis=1)

                if self.double_dqn:
                    next_q_online = self.online_net(s2, training=False)
                    next_actions = tf.argmax(next_q_online, axis=1)
                    next_q_target = self.target_net(s2, training=False)
                    next_q = tf.reduce_sum(next_q_target * tf.one_hot(next_actions, self.n_actions), axis=1)
                else:
                    next_q_target = self.target_net(s2, training=False)
                    next_q = tf.reduce_max(next_q_target, axis=1)

                target = r + (1.0 - d) * self.gamma * next_q
                loss = tf.reduce_mean(tf.keras.losses.Huber()(target, q_action))

        grads = tape.gradient(loss, self.online_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_net.trainable_variables))

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target()

        return float(loss.numpy())
