#!/usr/bin/env python3
"""Convert raw MobileArena HDF5 episodes into one Pi0.5-compatible LeRobot repo."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import cv2

POLICY_ROOT = Path(__file__).resolve().parents[2]
XPL_ROOT = POLICY_ROOT.parent.parent
OPENPI_SRC = POLICY_ROOT / "openpi" / "src"
for import_root in (XPL_ROOT.parent, OPENPI_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
from openpi import mobile_arena_layout as mobile_arena
from XPolicyLab.utils.process_data import decode_image_bit
from XPolicyLab.utils.benchmark_compat import mobile_arena_raw_root, mobile_arena_option

def _decode_instruction(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_instruction(value.item())
    return str(value).strip()

def _decode_image(value: object) -> np.ndarray:
    # XPolicyLab's helper uses cv2 internally and returns RGB. Keep that
    # convention unchanged for both LeRobot training and policy serving.
    image = np.asarray(decode_image_bit(value))
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
def _episode_data(path: Path):
    with h5py.File(path, "r") as f:
        state = mobile_arena.pack({key: f["state"][key][:] for key, _, _ in mobile_arena.KEYS}).astype(np.float32)
        action = mobile_arena.pack({key: f["action"][key][:] for key, _, _ in mobile_arena.KEYS}).astype(np.float32)
        instruction = _decode_instruction(f["instruction"][()])
        colors = {cam: f["vision"][cam]["colors"] for cam in mobile_arena.CAMERAS.values()}
        length = state.shape[0]
        if action.shape[0] != length:
            raise ValueError(f"{path}: state/action length mismatch {state.shape} {action.shape}")
        for cam, values in colors.items():
            if values.shape[0] != length:
                raise ValueError(f"{path}: {cam} length mismatch {values.shape[0]} != {length}")
        for index in range(length):
            images = {
                "cam_head": _decode_image(colors["cam_head"][index]),
                "cam_left_wrist": _decode_image(colors["cam_left_wrist"][index]),
                "cam_right_wrist": _decode_image(colors["cam_right_wrist"][index]),
            }
            yield state[index], action[index], images, instruction

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(mobile_arena_raw_root()))
    parser.add_argument("--repo-id", default=mobile_arena_option("REPO_ID", "miverse_data"))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dataset")
    args = parser.parse_args()

    home = Path(os.environ.get("HF_LEROBOT_HOME", str(POLICY_ROOT / "data"))).expanduser().resolve()
    os.environ["HF_LEROBOT_HOME"] = str(home)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    source_root = Path(args.source_root).expanduser().resolve()
    episode_files = sorted(source_root.rglob("*.hdf5"))
    if args.max_episodes is not None:
        episode_files = episode_files[: args.max_episodes]
    if not episode_files:
        raise FileNotFoundError(f"No MiVerse HDF5 files found under {source_root}")

    output_path = home / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {output_path}; pass --overwrite to replace it")
        shutil.rmtree(output_path)
    dim_names = [f"dim_{i:02d}" for i in range(mobile_arena.MODEL_DIM)]
    features = {
        "observation.state": {"dtype": "float32", "shape": (mobile_arena.MODEL_DIM,), "names": dim_names},
        "action": {"dtype": "float32", "shape": (mobile_arena.MODEL_DIM,), "names": dim_names},
    }
    for camera in mobile_arena.CAMERAS.values():
        features[f"observation.images.{camera}"] = {
            "dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(mobile_arena.FPS),
        robot_type=mobile_arena.ROBOT_TYPE,
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=1,
    )
    for episode_index, path in enumerate(episode_files):
        count = 0
        for state, action, images, instruction in _episode_data(path):
            dataset.add_frame({
                "observation.state": state,
                "action": action,
                "observation.images.cam_head": images["cam_head"],
                "observation.images.cam_left_wrist": images["cam_left_wrist"],
                "observation.images.cam_right_wrist": images["cam_right_wrist"],
                "task": instruction,
            })
            count += 1
        dataset.save_episode()
        dataset.hf_dataset = dataset.create_hf_dataset()
        print(f"[{episode_index + 1}/{len(episode_files)}] {path} frames={count}", flush=True)
    dataset.finalize()
    print(f"Wrote {len(episode_files)} episodes to {output_path}", flush=True)

if __name__ == "__main__":
    main()
