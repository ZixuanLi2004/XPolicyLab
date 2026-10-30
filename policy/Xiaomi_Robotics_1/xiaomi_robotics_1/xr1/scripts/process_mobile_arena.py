#!/usr/bin/env python3
"""Prepare raw MobileArena HDF5 with Xiaomi XR1's native JSON/video pipeline.

This follows the official XR1 ``process_data.py`` contract: JSON trajectory
files, decord-readable RGB MP4s, native instruction prompts, and per-horizon
60D statistics.  HDF5 ``action/*`` is always the target source; no action is
constructed from the next state.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import yaml
from tqdm import tqdm

XR1 = Path(__file__).resolve().parents[1]
if str(XR1) not in sys.path:
    sys.path.insert(0, str(XR1))
XPL_ROOT = Path(__file__).resolve().parents[6]
if str(XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(XPL_ROOT))
from XPolicyLab.utils.process_data import decode_image_bit  # noqa: E402
from XPolicyLab.utils.benchmark_compat import mobile_arena_raw_root
from mibot.utils.mobile_arena_adapter import (  # noqa: E402
    ACTION_RANGES,
    raw_fields_to_mobile_arena75,
    state75_to_xiaomi60,
    state75_to_xiaomi60_action,
)

CAMERAS = (("cam_head", "cam_head"), ("cam_left_wrist", "cam_left_wrist"), ("cam_right_wrist", "cam_right_wrist"))
RAW_NAMES = (
    "left_arm_joint_states", "right_arm_joint_states",
    "left_ee_joint_states", "right_ee_joint_states",
    "left_base_ee_poses", "right_base_ee_poses",
    "waist_joint_states", "leg_joint_states", "head_base_poses",
    "root_linear_velocity", "root_angular_velocity",
)
ACTION_LENGTH = 30
FPS_DEFAULT = 30.0


def text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray) and value.shape == ():
        return text(value.item())
    return str(value).strip()


def write_video(colors, out_path: Path, fps: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        os.fspath(out_path), fps=fps, codec="libx264", macro_block_size=1,
        ffmpeg_params=["-crf", "18", "-g", "1", "-pix_fmt", "yuv420p", "-threads", "1", "-loglevel", "error"],
    )
    try:
        for value in colors:
            # decode_image_bit is the XPolicyLab standard RGB decoder.
            writer.append_data(np.asarray(decode_image_bit(value), dtype=np.uint8))
    finally:
        writer.close()


def raw_arrays(group):
    return {name: np.asarray(group[name][:], dtype=np.float64) for name in RAW_NAMES}


def trajectory_json(src: Path, dst: Path, index: int, fps: float):
    with h5py.File(src, "r") as f:
        state = raw_arrays(f["state"])
        action = raw_arrays(f["action"])
        n = int(state[RAW_NAMES[0]].shape[0])
        if any(v.shape[0] != n for v in (*state.values(), *action.values())):
            raise ValueError(f"{src}: state/action fields have inconsistent lengths")
        instruction = text(f["instruction"][()])
        task = instruction.rstrip(".") + "."
        if "additional_info/frequency" in f:
            fps = float(f["additional_info/frequency"][()])
        out_data = dst / "data" / f"chunk-{index // 1000:04d}"
        out_video = dst / "videos" / f"chunk-{index // 1000:04d}"
        out_data.mkdir(parents=True, exist_ok=True)
        out_video.mkdir(parents=True, exist_ok=True)
        stem = f"episode_{index:07d}"
        video_paths = {}
        for src_cam, json_cam in CAMERAS:
            path = out_video / f"{stem}.{json_cam}.mp4"
            write_video(f["vision"][src_cam]["colors"], path, fps)
            video_paths[json_cam] = str(path)
        observations = {
            "ego": [{"path": video_paths["cam_head"], "start": 0, "crop_bbox": None}],
            "wrist_left": [{"path": video_paths["cam_left_wrist"], "start": 0, "crop_bbox": None}],
            "wrist_right": [{"path": video_paths["cam_right_wrist"], "start": 0, "crop_bbox": None}],
        }
        prompt = (
            "The following observations are captured from multiple views.\n"
            "# Ego View\n<image>\n# Left-Wrist View\n<image>\n"
            "# Right-Wrist View\n<image>\nGenerate robot actions for the task:\n" + task
        )
        payload = {
            "trajectory_type": "success",
            "time": src.stem,
            "num_frames": n,
            "instruction": {"general": [{"images": ["observations.ego", "observations.wrist_left", "observations.wrist_right"], "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]}]},
            "observations": observations,
            # Keep the original HDF5 key names so the loader's raw layout is explicit.
            "proprios": {k: v.tolist() for k, v in state.items()},
            "actions": {k: v.tolist() for k, v in action.items()},
        }
        json_path = out_data / f"{stem}.json"
        tmp = json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(json_path)
        return n, state, action


def read_source(src):
    """Read only the state/action groups of one HDF5 (no video, no JSON)."""
    with h5py.File(src, "r") as f:
        state = raw_arrays(f["state"])
        action = raw_arrays(f["action"])
    n = int(state[RAW_NAMES[0]].shape[0])
    if any(v.shape[0] != n for v in (*state.values(), *action.values())):
        raise ValueError(f"{src}: state/action fields have inconsistent lengths")
    return n, state, action


def process_one(job, stats_only=False):
    """Convert one HDF5 and return compact normalization accumulators."""
    index, src, dst = job
    try:
        if stats_only:
            n, state_raw, action_raw = read_source(src)
        else:
            n, state_raw, action_raw = trajectory_json(src, dst, index, FPS_DEFAULT)
        state75 = raw_fields_to_mobile_arena75(state_raw)
        # Critical: action75 comes from the HDF5 action/* group. It is never
        # synthesized from a shifted state sequence.
        action75 = raw_fields_to_mobile_arena75(action_raw)
        packed_state = state75_to_xiaomi60(state75).astype(np.float32)
        sums = np.zeros((ACTION_LENGTH, 60), dtype=np.float64)
        sums2 = np.zeros_like(sums)
        counts = np.zeros(ACTION_LENGTH, dtype=np.int64)
        for h in range(ACTION_LENGTH):
            if n <= h:
                continue
            current = state75[:-h] if h else state75
            target = action75[h:] if h else action75
            packed = state75_to_xiaomi60_action(current, target)
            sums[h] += packed.sum(axis=0)
            sums2[h] += np.square(packed).sum(axis=0)
            counts[h] += packed.shape[0]
        return index, n, packed_state, sums, sums2, counts, None
    except Exception as exc:
        return index, 0, None, None, None, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, default=mobile_arena_raw_root())
    ap.add_argument("--output-root", type=Path, default=XR1.parents[1] / "data" / "miverse_native")
    ap.add_argument("--config", type=Path, default=XR1 / "configs" / "data" / "mobile_arena_native.yaml")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stats-only", action="store_true", help="recompute mean/std/q01/q99 from the source HDF5 without re-encoding videos or JSON")
    args = ap.parse_args()
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    files = sorted(args.source_root.expanduser().rglob("*.hdf5"))
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise SystemExit(f"no HDF5 under {args.source_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    action_mask = np.zeros(60, dtype=bool)
    for lo, hi in ACTION_RANGES:
        action_mask[lo:hi] = True
    sums = np.zeros((ACTION_LENGTH, 60), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    counts = np.zeros(ACTION_LENGTH, dtype=np.int64)
    state_chunks = []
    manifest = [None] * len(files)
    failures = []
    total = 0
    jobs = [(i, src, args.output_root) for i, src in enumerate(files)]
    workers = max(1, min(int(args.workers), len(jobs)))
    print(f"source: {args.source_root} ({len(files)} HDF5)", flush=True)
    print(f"output: {args.output_root}; workers: {workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, job, args.stats_only) for job in jobs]
        with tqdm(total=len(futures), desc="MiVerse HDF5", unit="ep", smoothing=0.05) as bar:
            for future in as_completed(futures):
                index, n, packed_state, local_sums, local_sums2, local_counts, error = future.result()
                if error:
                    failures.append(f"episode {index}: {error}")
                else:
                    state_chunks.append(packed_state)
                    sums += local_sums
                    sums2 += local_sums2
                    counts += local_counts
                    total += n
                    manifest[index] = {"index": index, "source": str(files[index]), "frames": n}
                bar.update(1)
                bar.set_postfix(failed=len(failures), frames=total)
    if failures:
        print("conversion failures:", *failures[:20], sep="\n", flush=True)
        raise SystemExit(f"{len(failures)} episode conversions failed")
    denom = np.maximum(counts[:, None], 1)
    mean = sums / denom
    std = np.sqrt(np.maximum(sums2 / denom - mean * mean, 0.0))
    mean[:, ~action_mask] = 0.0
    std[:, ~action_mask] = 1.0
    std[:, action_mask] = np.maximum(std[:, action_mask], 1e-6)
    all_state = np.concatenate(state_chunks, axis=0)
    q01, q99 = np.quantile(all_state, [0.01, 0.99], axis=0)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    header = "# @package _global_\n"
    if args.stats_only and args.config.exists():
        previous = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        train = previous["data"]["params"]["train_datasets"]
        previous_mean = np.asarray(train["mean"], dtype=np.float64)
        previous_std = np.asarray(train["std"], dtype=np.float64)
        print("action mean drift (max abs):", float(np.abs(previous_mean - mean).max()), flush=True)
        print("action std  drift (max abs):", float(np.abs(previous_std - std).max()), flush=True)
        import shutil as _shutil
        _shutil.copy2(args.config, args.config.parent / (args.config.name + ".bak"))
        train["mean"], train["std"] = mean.tolist(), std.tolist()
        train["q01"], train["q99"] = [q01.tolist()], [q99.tolist()]
        args.config.write_text(header + yaml.safe_dump(previous, sort_keys=False), encoding="utf-8")
    else:
        cfg = {
            "#": "MiVerse raw HDF5 processed with native Xiaomi JSONDataset; HDF5 action group is used verbatim then mapped.",
            "data": {"type": "BaseDataModule", "params": {"type": "json", "max_steps": "$" + "{trainer.max_steps}", "num_workers": 8, "train_datasets": {
                "format": "json", "miverse_layout": True, "batch_size": 8, "action_length": ACTION_LENGTH,
                "paths": [str(args.output_root / "data")],
                "active_action_ranges": [list(r) for r in ACTION_RANGES],
                "mean": mean.tolist(), "std": std.tolist(), "q01": [q01.tolist()], "q99": [q99.tolist()],
            }}}
        }
        args.config.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if not args.stats_only:
        (args.output_root / "manifest.json").write_text(json.dumps({"source_root": str(args.source_root), "episodes": manifest, "frames": total, "action_ranges": [list(r) for r in ACTION_RANGES]}, indent=2), encoding="utf-8")

    print(f"wrote {len(files)} episodes / {total} frames to {args.output_root}", flush=True)
    print(f"config: {args.config}; active dims: {int(action_mask.sum())}", flush=True)


if __name__ == "__main__":
    main()
