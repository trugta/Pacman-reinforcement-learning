import argparse
import glob
import json
import os
import re
from typing import Any


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")


def _extract_ckpt_step(name: str) -> int | None:
    match = re.search(r"ckpt-(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def _read_text_checkpoint_file(path: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "model_checkpoint_path": None,
        "all_model_checkpoint_paths": [],
        "latest_ckpt_step": None,
        "max_ckpt_step": None,
    }
    if not os.path.exists(path):
        return data

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("model_checkpoint_path:"):
            value = line.split(":", 1)[1].strip().strip('"')
            data["model_checkpoint_path"] = value
        elif line.startswith("all_model_checkpoint_paths:"):
            value = line.split(":", 1)[1].strip().strip('"')
            data["all_model_checkpoint_paths"].append(value)

    if data["model_checkpoint_path"]:
        data["latest_ckpt_step"] = _extract_ckpt_step(data["model_checkpoint_path"])

    steps = [
        _extract_ckpt_step(name)
        for name in data["all_model_checkpoint_paths"]
        if _extract_ckpt_step(name) is not None
    ]
    if steps:
        data["max_ckpt_step"] = max(steps)

    return data


def _read_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _list_ckpt_index_steps(checkpoint_dir: str) -> list[int]:
    pattern = os.path.join(checkpoint_dir, "ckpt-*.index")
    files = glob.glob(pattern)
    steps: list[int] = []
    for path in files:
        step = _extract_ckpt_step(os.path.basename(path))
        if step is not None:
            steps.append(step)
    return sorted(set(steps))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify checkpoint status and print latest known training step."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing checkpoint artifacts (default: checkpoints)",
    )
    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    print("=" * 68)
    print("Checkpoint Verification")
    print("=" * 68)
    print(f"Checkpoint dir: {checkpoint_dir}")

    if not os.path.isdir(checkpoint_dir):
        print("Status: checkpoint directory does not exist.")
        return 1

    checkpoint_txt_path = os.path.join(checkpoint_dir, "checkpoint")
    checkpoint_h5_path = os.path.join(checkpoint_dir, "checkpoint.h5")
    metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")

    txt_info = _read_text_checkpoint_file(checkpoint_txt_path)
    metadata = _read_json(metadata_path)
    ckpt_index_steps = _list_ckpt_index_steps(checkpoint_dir)

    print(f"Has checkpoint.h5: {os.path.exists(checkpoint_h5_path)}")
    print(f"Has checkpoint file: {os.path.exists(checkpoint_txt_path)}")
    print(f"Has checkpoint_metadata.json: {metadata is not None}")
    print("-")

    if txt_info["model_checkpoint_path"]:
        print(f"checkpoint -> latest model_checkpoint_path: {txt_info['model_checkpoint_path']}")
    if txt_info["latest_ckpt_step"] is not None:
        print(f"checkpoint -> latest ckpt step: {txt_info['latest_ckpt_step']}")
    if txt_info["max_ckpt_step"] is not None:
        print(f"checkpoint -> max listed ckpt step: {txt_info['max_ckpt_step']}")

    if ckpt_index_steps:
        print(f"ckpt-*.index files found: {len(ckpt_index_steps)}")
        print(f"ckpt-*.index step range: {ckpt_index_steps[0]} .. {ckpt_index_steps[-1]}")
    else:
        print("No ckpt-*.index files found.")

    train_steps = None
    env_steps = None
    if metadata:
        train_steps = metadata.get("train_steps")
        env_steps = metadata.get("env_steps")
        print("-")
        print(f"metadata -> train_steps: {train_steps}")
        if env_steps is not None:
            print(f"metadata -> env_steps: {env_steps}")
        else:
            print("metadata -> env_steps: <missing>")

    print("-")
    print("Best progress signal:")
    if env_steps is not None:
        print(f"  Reached env step: {env_steps}")
    elif train_steps is not None:
        print(f"  Reached train step: {train_steps}")
    elif txt_info["latest_ckpt_step"] is not None:
        print(f"  Latest TensorFlow checkpoint suffix: ckpt-{txt_info['latest_ckpt_step']}")
    elif ckpt_index_steps:
        print(f"  Highest ckpt-*.index suffix: ckpt-{ckpt_index_steps[-1]}")
    else:
        print("  Could not determine training step from available files.")

    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 