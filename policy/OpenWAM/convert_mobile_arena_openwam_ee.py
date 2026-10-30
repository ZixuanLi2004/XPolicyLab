#!/usr/bin/env python3
"""Convert the raw MobileArena HDF5 release to an OpenWAM LeRobot-v3 bucket.

The HDF5 state/action fields are the source of truth.  Each row is packed as
45 raw dimensions:

    L EE xyz+rot6d (9), L dex hand (12), R EE xyz+rot6d (9),
    R dex hand (12), base vx/vy/wz (3).

The OpenWAM reader scatters these dimensions into its 80-D unified space with
the map in ``meta/openwam_layout.json``.  Image bytes are decoded only with
XPolicyLab's ``decode_image_bit`` (which returns RGB) and are written to MP4
without a channel swap.

Conversion is restartable.  A worker writes parquet/video files to temporary
names and atomically creates ``.done/episode_*.json`` only after all files and
statistics are complete.  Re-running skips completed episodes and rebuilds
the metadata from the done markers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np

FPS = 30
RAW_DIM = 45
UNIFY_DIM = 80
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
VIDEO_KEYS = tuple(f"observation.images.{c}" for c in CAMERAS)
FILES_PER_CHUNK = 1000

POLICY_DIR = Path(__file__).resolve().parent
OUT_DEFAULT = POLICY_DIR / "data" / "miverse_ee"
XPL_PARENT = POLICY_DIR.parents[2]


def _episode_paths(source: Path):
    tasks = sorted(p for p in source.iterdir() if p.is_dir())
    jobs = []
    for task_index, task_dir in enumerate(tasks):
        data_dir = task_dir / "miverse" / "data"
        for local_index, path in enumerate(sorted(data_dir.glob("episode_*.hdf5"))):
            jobs.append((len(jobs), task_index, task_dir.name, local_index, str(path)))
    return jobs, tasks


def _decode_instruction(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.ndarray) and x.shape == ():
        return _decode_instruction(x.item())
    return str(x)


def _quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    """MobileArena stores [x,y,z,qw,qx,qy,qz]; return first two matrix columns."""
    q = np.asarray(q, dtype=np.float32)
    w, x, y, z = (q[..., 3], q[..., 4], q[..., 5], q[..., 6])
    n = np.sqrt(w*w + x*x + y*y + z*z)
    n = np.maximum(n, 1e-8)
    w, x, y, z = w/n, x/n, y/n, z/n
    c0 = np.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)], axis=-1)
    c1 = np.stack([2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)], axis=-1)
    return np.concatenate([c0, c1], axis=-1).astype(np.float32)


def _pack_group(g) -> np.ndarray:
    left_pose = np.concatenate([np.asarray(g["left_base_ee_poses"][:, :3]), _quat_wxyz_to_rot6d(g["left_base_ee_poses"])], axis=-1)
    right_pose = np.concatenate([np.asarray(g["right_base_ee_poses"][:, :3]), _quat_wxyz_to_rot6d(g["right_base_ee_poses"])], axis=-1)
    left_hand = np.asarray(g["left_ee_joint_states"])
    right_hand = np.asarray(g["right_ee_joint_states"])
    base = np.concatenate([np.asarray(g["root_linear_velocity"])[:, :2], np.asarray(g["root_angular_velocity"])[:, 2:3]], axis=-1)
    out = np.concatenate([left_pose, left_hand, right_pose, right_hand, base], axis=-1)
    if out.shape[-1] != RAW_DIM:
        raise ValueError(f"packed HDF5 layout has {out.shape[-1]} dims, expected {RAW_DIM}")
    return out.astype(np.float32, copy=False)


def _stats(x: np.ndarray) -> dict:
    return {"count": int(x.shape[0]), "sum": x.sum(0).tolist(), "sumsq": (x*x).sum(0).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist()}


def _encode_video(colors, dest: Path, size: int, crf: int) -> int:
    import av
    import cv2
    sys.path.insert(0, str(XPL_PARENT))
    from XPolicyLab.utils.process_data import decode_image_bit

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(tmp), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = size
    stream.height = size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "veryfast", "bf": "0", "g": "8", "threads": "2"}
    written = 0
    for raw in colors:
        rgb = np.asarray(decode_image_bit(raw))
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"decoded {dest.name} frame has shape {rgb.shape}")
        if rgb.shape[0] != size or rgb.shape[1] != size:
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = written
        frame.time_base = Fraction(1, FPS)
        for packet in stream.encode(frame):
            container.mux(packet)
        written += 1
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    os.replace(tmp, dest)
    return written


def _write_data(path: Path, state: np.ndarray, action: np.ndarray, sid: int, source_episode_index: int, task_index: int, offset: int):
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = state.shape[0]
    fixed = lambda x: pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1), type=pa.float32()), RAW_DIM)
    table = pa.table({
        "observation.state": fixed(state), "action": fixed(action),
        "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.full(n, sid, dtype=np.int64)),
        "index": pa.array(np.arange(offset, offset + n, dtype=np.int64)),
        "task_index": pa.array(np.full(n, task_index, dtype=np.int64)),
        "source_episode_index": pa.array(np.full(n, source_episode_index, dtype=np.int64)),
    })
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def _convert_one(job, out_str: str, size: int, crf: int, force: bool = False):
    sid, task_index, task_name, local_index, source_str = job
    import h5py
    out = Path(out_str)
    c, f = divmod(sid, FILES_PER_CHUNK)
    done = out / ".done" / f"episode_{sid:07d}.json"
    if done.exists() and not force:
        return {"sid": sid, "status": "skip"}
    data_path = out / "data" / f"chunk-{c:03d}" / f"file-{f:03d}.parquet"
    video_paths = {cam: out / "videos" / f"observation.images.{cam}" / f"chunk-{c:03d}" / f"file-{f:03d}.mp4" for cam in CAMERAS}
    done.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(source_str, "r") as h5:
            state = _pack_group(h5["state"])
            action = _pack_group(h5["action"])
            if state.shape != action.shape or state.shape[0] < 2:
                raise ValueError(f"state/action shape mismatch: {state.shape} vs {action.shape}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"non-finite state/action in {source_str}")
            instruction = _decode_instruction(h5["instruction"][()])
            n = int(state.shape[0])
            _write_data(data_path, state, action, sid, local_index, task_index, 0)
            for cam, vp in video_paths.items():
                written = _encode_video(h5["vision"][cam]["colors"], vp, size, crf)
                if written != n:
                    raise ValueError(f"{source_str}: {cam} wrote {written} frames, expected {n}")
        payload = {
            "episode_index": sid, "task_index": task_index, "task_name": task_name,
            "source_episode_index": local_index, "source": source_str,
            "instruction": instruction, "length": n,
            "stats_state": _stats(state), "stats_action": _stats(action),
            "chunk_index": c, "file_index": f,
        }
        tmp = done.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        os.replace(tmp, done)
        return {"sid": sid, "status": "done", "length": n}
    except Exception:
        for p in [data_path, *video_paths.values()]:
            try: p.unlink()
            except FileNotFoundError: pass
        raise


def _merge_stats(items, key):
    count = 0; sm = np.zeros(RAW_DIM); ss = np.zeros(RAW_DIM); mn = np.full(RAW_DIM, np.inf); mx = np.full(RAW_DIM, -np.inf)
    for x in items:
        s = x[key]; count += int(s["count"]); sm += np.asarray(s["sum"]); ss += np.asarray(s["sumsq"]); mn = np.minimum(mn, s["min"]); mx = np.maximum(mx, s["max"])
    mean = sm / max(count, 1); var = np.maximum(ss / max(count, 1) - mean*mean, 1e-12)
    # Rot6D columns are a coupled rotation basis.  Keep them in their native
    # [-1, 1] representation instead of independently affine-normalizing them.
    rot = list(range(3, 9)) + list(range(24, 30))
    mean[rot] = 0.0; var[rot] = 1.0; mn[rot] = -1.0; mx[rot] = 1.0
    return {"count": count, "mean": mean.tolist(), "std": np.sqrt(var).tolist(), "min": mn.tolist(), "max": mx.tolist(), "q01": mn.tolist(), "q99": mx.tolist()}


def _rebuild_metadata(out: Path, tasks: list[Path], size: int):
    import pyarrow as pa
    import pyarrow.parquet as pq
    markers = [json.loads(p.read_text()) for p in sorted((out / ".done").glob("episode_*.json"))]
    if not markers:
        return 0
    markers.sort(key=lambda x: int(x["episode_index"]))
    total = 0; eps = []; task_text = {}
    for m in markers:
        sid = int(m["episode_index"]); n = int(m["length"]); c = int(m["chunk_index"]); f = int(m["file_index"])
        start = total; total += n; ti = int(m["task_index"]); task_text[ti] = m["instruction"]
        row = {"episode_index": sid, "tasks": [m["instruction"]], "length": n, "dataset_from_index": start, "dataset_to_index": total, "data/chunk_index": c, "data/file_index": f, "source_episode_index": int(m["source_episode_index"])}
        for vk in VIDEO_KEYS:
            row[f"videos/{vk}/chunk_index"] = c; row[f"videos/{vk}/file_index"] = f; row[f"videos/{vk}/from_timestamp"] = 0.0; row[f"videos/{vk}/to_timestamp"] = (n - 1) / FPS
        eps.append(row)
    (out / "meta/episodes").mkdir(parents=True, exist_ok=True)
    (out / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(eps), out / "meta/episodes/chunk-000/file-000.parquet")
    tasks_df = __import__("pandas").DataFrame({"task_index": sorted(task_text), "task": [task_text[k] for k in sorted(task_text)]}).set_index("task")
    tasks_df.to_parquet(out / "meta/tasks.parquet")
    act = _merge_stats(markers, "stats_action"); sta = _merge_stats(markers, "stats_state")
    (out / "meta/stats.json").write_text(json.dumps({"action": act, "observation.state": sta}, ensure_ascii=False, indent=2))
    layout = {
        "raw_dim": RAW_DIM, "unify_dim": UNIFY_DIM, "fps": FPS,
        "raw_order": ["left_ee_xyz_rot6d(9)", "left_hand(12)", "right_ee_xyz_rot6d(9)", "right_hand(12)", "base_vx_vy_wz(3)"],
        "unify_map": [[0, 0, 8], [9, 10, 21], [21, 34, 42], [30, 44, 55], [42, 68, 70]],
        "masked_unified_dims": [9, *range(22, 34), 43, *range(56, 68), *range(71, 80)],
        "pose_source": "HDF5 [x,y,z,qw,qx,qy,qz] -> xyz + first two rotation-matrix columns (rot6d)",
        "image_decode": "XPolicyLab.utils.process_data.decode_image_bit -> RGB; no channel swap",
    }
    (out / "meta/openwam_layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2))
    shape = [size, size, 3]
    features = {k: {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"], "info": {"video.height": size, "video.width": size, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.fps": FPS, "video.channels": 3}} for k in VIDEO_KEYS}
    features.update({"observation.state": {"dtype": "float32", "shape": [RAW_DIM]}, "action": {"dtype": "float32", "shape": [RAW_DIM]}, "timestamp": {"dtype": "float32", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]}, "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]}, "task_index": {"dtype": "int64", "shape": [1]}, "source_episode_index": {"dtype": "int64", "shape": [1]}})
    info = {"codebase_version": "v3.0", "robot_type": "m92uw", "total_episodes": len(markers), "total_frames": total, "total_tasks": len(task_text), "fps": FPS, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet", "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": features, "layout": layout, "total_chunks": 1, "chunks_size": FILES_PER_CHUNK, "total_videos": len(markers)*len(VIDEO_KEYS)}
    (out / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    (out / "meta/embodiment.json").write_text(json.dumps({"name": "m92uw", "action_dim": RAW_DIM, "state_dim": RAW_DIM, "openwam_unified_dim": UNIFY_DIM, "source": "MiVerse HDF5"}, indent=2))
    return len(markers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="MiVerse raw HDF5 root containing task directories")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs, tasks = _episode_paths(args.src)
    if args.max_episodes:
        jobs = jobs[: args.max_episodes]
    if not args.metadata_only:
        print(f"jobs={len(jobs)} workers={args.workers} out={args.out}", flush=True)
        errors = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_convert_one, j, str(args.out), args.size, args.crf, args.force) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    result = fut.result()
                    if i % 10 == 0 or result.get("status") == "done": print(f"[{i}/{len(futs)}] {result}", flush=True)
                except Exception as exc:
                    errors += 1; print(f"ERROR {i}: {exc}", flush=True); traceback.print_exc()
        if errors:
            raise SystemExit(f"{errors} episodes failed; completed markers are retained for restart")
    n = _rebuild_metadata(args.out, tasks, args.size)
    print(f"metadata episodes={n}", flush=True)


if __name__ == "__main__":
    main()
