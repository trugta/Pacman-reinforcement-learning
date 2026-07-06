# DQN / Double DQN for ALE/MsPacman-v5

This repository contains a DQN / Double DQN implementation to train an agent to play Ms. Pacman using `gymnasium` + `ale-py` and `tensorflow` (Keras).

Project layout (key files):
- `wrappers.py` — Gymnasium observation wrappers (`PreprocessFrame`, `FrameStack`).
- `buffer.py` — `ReplayBuffer` implementation.
- `model.py` — CNN Q-network builder (`build_q_network`).
- `agent.py` — `DQNAgent` class (action selection, memory, training step).
- `train.py` — training entrypoint and `train_loop`.
- `test.py` — load saved weights and render agent playing.
- `requirements.txt` — Python dependencies.
- `README_DQN.md` — this file.

Quick setup (Windows):

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train (example small run):

```powershell
python train.py
```

To customize training parameters, edit `train.py` or call `train_loop` from Python:

```python
from train import train_loop
train_loop(num_episodes=500, eval_every=50)
```

4. Evaluate / play using a saved model:

```powershell
python test.py
# or from Python (you can pass the base name; the loader will add the correct suffix)
from test import play
play('best_mspacman', episodes=1)
```

Notes and tips:
- `wrappers.py` uses `opencv-python` when available; `Pillow` is used as a fallback for image resizing.
- For GPU-enabled TensorFlow on Windows, prefer using WSL2 or the TensorFlow-DirectML plugin — native TensorFlow >=2.11 has limited GPU support on Windows.
- Training Atari agents is compute-intensive; for development and debugging, use small `num_episodes`, smaller replay buffers, and lower `target_update_freq`.
- Saved model weights are written by default to `best_mspacman.weights.h5` by the training loop.

If you want, I can also add example configuration flags (CLI) to `train.py` for easy parameterization.

**CLI Examples**

Train with defaults:

```powershell
python train.py
```

Train with custom settings:

```powershell
python train.py --episodes 500 --eval-every 50 --lr 0.00025 --batch-size 64 --eps-decay 500000
```

Checkpointing (resumeable training)

```powershell
# start training and write full checkpoints (model + optimizer) into 'checkpoints/'
python train.py --checkpoint-dir checkpoints

# resume training from latest checkpoint in 'checkpoints/'
python train.py --checkpoint-dir checkpoints --resume-checkpoint
```

Run the tester to play a saved model:

```powershell
python test.py
```

