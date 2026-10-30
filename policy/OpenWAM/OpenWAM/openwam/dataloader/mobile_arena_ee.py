"""OpenWAM reader for the MobileArena EE/raw-45 LeRobot-v3 bucket."""

from __future__ import annotations

import json
import os
import socket
import uuid
from typing import Any, ClassVar, Optional, Tuple

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats, STAT_KEYS


def _column_matrix(win, column: str, dim: int) -> np.ndarray:
    rows = [np.asarray(x, dtype=np.float32) for x in win[column].tolist()]
    out = np.stack(rows, axis=0).astype(np.float32, copy=False)
    if out.ndim != 2 or out.shape[-1] != dim:
        raise ValueError(f"MobileArena {column} must be (T,{dim}), got {out.shape}")
    return out


class MobileArenaEEDataset(LeRobotV3Reader):
    """MobileArena HDF5 EE state/action packed as raw 45D, unified to OpenWAM 80D."""

    DATASET_NAME = "MobileArenaEE"
    ACTION_DIM = 45
    STATS_DIM = 45
    DEFAULT_NORMALIZE_MODE = "min-max"
    STATS_FILENAME = None
    DEPLOY_ACTION_MODE = "mobile_arena_ee"
    HEAD_CAMERA = "observation.images.cam_head"
    LEFT_WRIST_CAMERA = "observation.images.cam_left_wrist"
    RIGHT_WRIST_CAMERA = "observation.images.cam_right_wrist"
    NEEDED_COLS: ClassVar[Tuple[str, ...]] = (
        "task_index", "observation.state", "action", "timestamp", "frame_index",
        "episode_index", "index", "source_episode_index",
    )

    def _load_stats(self, info: dict) -> Optional[dict]:
        if not self._normalize_mode or str(self._normalize_mode).lower() in ("none", "null"):
            self._state_normalization_stats = None
            return None
        path = self._dataset_dir / "meta" / "stats.json"
        raw = json.loads(path.read_text())
        action_raw = raw.get("action", raw)
        state_raw = raw.get("observation.state", raw)
        action_stats = materialize_eef_stats(
            action_raw, self._normalize_mode, dim=self._raw_action_dim,
            strict_minmax=False, source_hint=str(path), force_rot6d_identity=False,
        )
        self._state_normalization_stats = materialize_eef_stats(
            state_raw, self._normalize_mode, dim=self._raw_action_dim,
            strict_minmax=False, source_hint=str(path), force_rot6d_identity=False,
        )
        # Deployment denormalizes the model's unified output back into this
        # raw 45D schema, so keep the artifact in raw (pre-scatter) space.
        # Store both directions together; state and command ranges differ.
        payload = {
            self.DEPLOY_ACTION_MODE: {k: np.asarray(action_stats[k], dtype=np.float32) for k in STAT_KEYS},
            self.DEPLOY_ACTION_MODE + "_state": {
                k: np.asarray(self._state_normalization_stats[k], dtype=np.float32) for k in STAT_KEYS
            },
        }
        dest = self._dataset_dir / "meta" / "normalization_stats.npy"
        tmp = dest.with_name(f".mobile_arena_stats.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}.npy")
        try:
            np.save(tmp, payload, allow_pickle=True)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        self.normalization_stats_path = str(dest)
        return action_stats

    def _action_20d(self, win) -> Optional[np.ndarray]:
        action = _column_matrix(win, "action", self._raw_action_dim)
        return apply_normalization(action, self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win) -> Optional[np.ndarray]:
        state = _column_matrix(win, "observation.state", self._raw_action_dim)
        state = apply_normalization(state, self._state_normalization_stats, self._normalize_mode)
        return state[:1]


__all__ = ["MobileArenaEEDataset"]
