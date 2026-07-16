# DQN / Double DQN for ALE/MsPacman-v5

This repository contains a DQN / Double DQN implementation to train an agent to play Ms. Pacman using `gymnasium` + `ale-py` and `tensorflow` (Keras).

Project layout (key files):
- `src/wrappers.py` — Gymnasium observation wrappers (`PreprocessFrame`, `FrameStack`).
- `src/model.py` — CNN Q-network builder (`build_q_network`).
- `src/agent.py` — `DQNAgent` class (action selection, replay buffer, training step).
- `train.py` — training entrypoint and `train_loop`.
- `test.py` — load a saved checkpoint and evaluate/render the agent.
- `checkpoints/` — model weights, optimizer states, and checkpoint metadata.
- `logs/` — TensorBoard or training metric logs.
- `requirements.txt` — Python dependencies.
- `README.md` — this file.
- `CHANGELOG.md` — history of fixes and known issues.

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

4. Evaluate / play using a saved checkpoint:

```powershell
python test.py --episodes 1
# or from Python
from test import play
play(episodes=1, checkpoint_dir='checkpoints')
```

Notes and tips:
- `wrappers.py` uses `opencv-python` when available; `Pillow` is used as a fallback for image resizing.
- For GPU-enabled TensorFlow on Windows, prefer using WSL2 or the TensorFlow-DirectML plugin — native TensorFlow >=2.11 has limited GPU support on Windows.
- Training Atari agents is compute-intensive; for development and debugging, use small `num_episodes`, smaller replay buffers, and lower `target_update_freq`.
- Saved model weights and checkpoints are written under `checkpoints/` by default.

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

python train.py --steps 1000000 --lr 0.0001 --eps-decay 200000 --log-every 5000 --checkpoint-every 10000
# resume training from a specific checkpoint file
python train.py --steps <remaining_steps> --resume-checkpoint checkpoints/checkpoint.h5 --lr 0.0001 --eps-decay 200000 --log-every 5000 --checkpoint-every 10000

Checkpointing (resumeable training)

```powershell
# start training and write full checkpoints (model + optimizer) into 'checkpoints/'
python train.py --checkpoint-dir checkpoints

# resume training from a specific checkpoint file
python train.py --checkpoint-dir checkpoints --resume-checkpoint checkpoints/checkpoint.h5
```

Run the tester to play a saved model:

```powershell
python test.py --episodes 10 --render
```