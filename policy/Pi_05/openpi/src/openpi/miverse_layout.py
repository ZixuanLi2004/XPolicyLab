"""MiVerse m92uw 45D layout used by Pi0.5 training and deployment.

The model vector intentionally contains only the controllable MiVerse degrees of
freedom requested for this run:

    0:7    left arm joints
    7:14   right arm joints
    14:26  left dexterous hand
    26:38  right dexterous hand
    38:40  waist joints
    40:42  leg joints
    42:45  mobile base [vx, vy, wz]

The source HDF5 still contains all original MiVerse keys.  ``pack`` ignores
uncontrolled pose/head/root components, and ``unpack`` reconstructs the two
runtime root-velocity arrays with the omitted components set to zero.  This
keeps the XPolicyLab runtime key contract while ensuring the learned head is
exactly 45D.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

EMBODIMENT_TAG = "miverse_m92uw_m45"
ROBOT_TYPE = "m92uw"
FPS = 30.0

# All source keys are retained so the HDF5 converter can enumerate the raw file.
KEYS: tuple[tuple[str, int, str], ...] = (
    ("left_arm_joint_states", 7, "vec"),
    ("right_arm_joint_states", 7, "vec"),
    ("left_ee_joint_states", 12, "vec"),
    ("right_ee_joint_states", 12, "vec"),
    ("left_base_ee_poses", 7, "pose"),
    ("right_base_ee_poses", 7, "pose"),
    ("waist_joint_states", 2, "vec"),
    ("leg_joint_states", 2, "vec"),
    ("head_base_poses", 7, "pose"),
    ("root_linear_velocity", 3, "vec"),
    ("root_angular_velocity", 3, "vec"),
)

# Only these entries are represented in the model vector.  The root controls
# are handled specially below because only vx, vy, and wz are effective.
MODEL_KEYS: tuple[tuple[str, int, str], ...] = (
    KEYS[0], KEYS[1], KEYS[2], KEYS[3], KEYS[6], KEYS[7],
)
BASE_START = sum(dim for _, dim, _ in MODEL_KEYS)
MODEL_DIM = BASE_START + 3  # 42 + [vx, vy, wz]
RAW_DIM = sum(dim for _, dim, _ in KEYS)

CAMERAS: dict[str, str] = {
    "primary": "cam_head",
    "wrist_left": "cam_left_wrist",
    "wrist_right": "cam_right_wrist",
}

KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "head_base_poses": ("base_head_poses",),
}
CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    "cam_head": ("head_infra1", "head_infra", "head_camera"),
    "cam_left_wrist": ("left_camera",),
    "cam_right_wrist": ("right_camera",),
}


def singular(key: str) -> str:
    if key.endswith("_states"):
        return key[: -len("_states")] + "_state"
    if key.endswith("_poses"):
        return key[: -len("_poses")] + "_pose"
    return key


def _lookup(values: Mapping[str, object], key: str):
    for cand in (key, singular(key), *KEY_ALIASES.get(key, ()),
                 *(singular(a) for a in KEY_ALIASES.get(key, ()) )):
        for name in (cand, f"state.{cand}", f"state/{cand}"):
            if name in values:
                return values[name]
    return None


def _lead_from_values(values: Mapping[str, object]) -> tuple[int, ...]:
    for key, dim, _ in (*MODEL_KEYS, KEYS[9], KEYS[10]):
        raw = _lookup(values, key)
        if raw is not None:
            arr = np.asarray(raw)
            if arr.ndim == 0 or arr.shape[-1] != dim:
                raise ValueError(f"{key}: expected last dim {dim}, got {arr.shape}")
            return arr.shape[:-1]
    return ()


def _require_or_zero(values: Mapping[str, object], key: str, dim: int,
                     lead: tuple[int, ...], *, allow_missing: bool) -> np.ndarray:
    raw = _lookup(values, key)
    if raw is None:
        if not allow_missing:
            raise KeyError(f"missing MiVerse key {key!r} (or {singular(key)!r})")
        return np.zeros((*lead, dim), dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != dim:
        raise ValueError(f"{key}: expected last dim {dim}, got {arr.shape}")
    return arr.reshape((*lead, dim))


def pack(values: Mapping[str, object], *, allow_missing: bool = False) -> np.ndarray:
    """Pack raw HDF5/runtime values into the 45D training/deployment vector."""
    lead = _lead_from_values(values)
    parts = []
    for key, dim, kind in MODEL_KEYS:
        if kind != "vec":
            raise RuntimeError(f"unexpected non-vector key in 45D layout: {key}")
        parts.append(_require_or_zero(values, key, dim, lead, allow_missing=allow_missing))

    linear = _require_or_zero(values, "root_linear_velocity", 3, lead, allow_missing=allow_missing)
    angular = _require_or_zero(values, "root_angular_velocity", 3, lead, allow_missing=allow_missing)
    parts.extend((linear[..., :2], angular[..., 2:3]))
    return np.concatenate(parts, axis=-1).astype(np.float64, copy=False)


def unpack(vec: np.ndarray, *, key_style: str = "singular") -> dict[str, np.ndarray]:
    """Unpack a 45D vector into runtime action keys.

    The returned root arrays retain their original 3D shapes.  ``vz``, ``wx``
    and ``wy`` are explicitly zero because they are not modelled in 45D.
    """
    vec = np.asarray(vec, dtype=np.float64)
    if vec.shape[-1] != MODEL_DIM:
        raise ValueError(f"expected last dim {MODEL_DIM}, got {vec.shape}")
    out: dict[str, np.ndarray] = {}
    for key, _, _ in MODEL_KEYS:
        start = next(s for s in _slots() if s[0] == key)[1]
        end = next(s for s in _slots() if s[0] == key)[2]
        name = singular(key) if key_style == "singular" else key
        out[name] = vec[..., start:end].astype(np.float32)

    linear = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
    angular = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
    linear[..., :2] = vec[..., BASE_START : BASE_START + 2]
    angular[..., 2] = vec[..., BASE_START + 2]
    out["root_linear_velocity"] = linear
    out["root_angular_velocity"] = angular
    return out


def _slots() -> tuple[tuple[str, int, int], ...]:
    slots = []
    cursor = 0
    for key, dim, _ in MODEL_KEYS:
        slots.append((key, cursor, cursor + dim))
        cursor += dim
    return tuple(slots)


SLOTS = _slots()


def contract(variant: str = "m45") -> dict:
    return {
        "variant": variant,
        "model_width": MODEL_DIM,
        "embodiment_tag": EMBODIMENT_TAG,
        "robot_type": ROBOT_TYPE,
        "fps": FPS,
        "model_dim": MODEL_DIM,
        "raw_dim": RAW_DIM,
        "cameras": dict(CAMERAS),
        "keys": [
            {
                "key": key,
                "runtime_key": singular(key),
                "raw_dim": dim,
                "kind": kind,
                "model_slice": [next(s for s in SLOTS if s[0] == key)[1], next(s for s in SLOTS if s[0] == key)[2]],
            }
            for key, dim, kind in MODEL_KEYS
        ] + [{
            "key": "mobile_base_effective",
            "runtime_keys": ["root_linear_velocity", "root_angular_velocity"],
            "raw_components": ["vx", "vy", "wz"],
            "model_slice": [BASE_START, MODEL_DIM],
        }],
        "pose_format": None,
    }
