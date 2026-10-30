"""MobileArena EE45 boundary; model-side unified mapping stays in the checkpoint."""
from collections.abc import Mapping

import numpy as np


RAW_DIM = 45
NATIVE_DIMS = {
    "left_base_ee_pose": 7,
    "left_ee_joint_state": 12,
    "right_base_ee_pose": 7,
    "right_ee_joint_state": 12,
    "root_linear_velocity": 3,
    "root_angular_velocity": 3,
}


_EPS = 1e-8


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt on the first two columns."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS, None)
    a2_proj = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_proj / np.clip(np.linalg.norm(a2_proj, axis=-1, keepdims=True), _EPS, None)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)

def matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6), the inverse of rot6d_to_matrix up to gauge."""
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)

def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-first unit quaternion -> (..., 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), _EPS, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )

def matrix_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) scalar-first unit quaternion with w >= 0.

    Shepperd's method: pick the largest of (trace, diagonal) as the pivot so the
    square root never sees a near-zero argument.
    """
    m = np.asarray(mat, dtype=np.float64)
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    cand = np.stack([trace, m00, m11, m22], axis=-1)
    pivot = np.argmax(cand, axis=-1)
    q = np.zeros(m.shape[:-2] + (4,), dtype=np.float64)

    def _set(sel, w, x, y, z):
        q[sel, 0], q[sel, 1], q[sel, 2], q[sel, 3] = w[sel], x[sel], y[sel], z[sel]

    s0 = np.sqrt(np.clip(1.0 + trace, _EPS, None)) * 2.0
    _set(pivot == 0, 0.25 * s0, (m[..., 2, 1] - m[..., 1, 2]) / s0,
         (m[..., 0, 2] - m[..., 2, 0]) / s0, (m[..., 1, 0] - m[..., 0, 1]) / s0)
    s1 = np.sqrt(np.clip(1.0 + m00 - m11 - m22, _EPS, None)) * 2.0
    _set(pivot == 1, (m[..., 2, 1] - m[..., 1, 2]) / s1, 0.25 * s1,
         (m[..., 0, 1] + m[..., 1, 0]) / s1, (m[..., 0, 2] + m[..., 2, 0]) / s1)
    s2 = np.sqrt(np.clip(1.0 - m00 + m11 - m22, _EPS, None)) * 2.0
    _set(pivot == 2, (m[..., 0, 2] - m[..., 2, 0]) / s2, (m[..., 0, 1] + m[..., 1, 0]) / s2,
         0.25 * s2, (m[..., 1, 2] + m[..., 2, 1]) / s2)
    s3 = np.sqrt(np.clip(1.0 - m00 - m11 + m22, _EPS, None)) * 2.0
    _set(pivot == 3, (m[..., 1, 0] - m[..., 0, 1]) / s3, (m[..., 0, 2] + m[..., 2, 0]) / s3,
         (m[..., 1, 2] + m[..., 2, 1]) / s3, 0.25 * s3)

    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), _EPS, None)
    return np.where(q[..., :1] < 0.0, -q, q)

def pose_to_model(pose):
    pose = np.asarray(pose, dtype=np.float64)
    r6 = matrix_to_rot6d(quat_wxyz_to_matrix(pose[..., 3:7]))
    return np.concatenate([pose[..., :3], r6], axis=-1)


def model_to_pose(vec):
    vec = np.asarray(vec, dtype=np.float64)
    q = matrix_to_quat_wxyz(rot6d_to_matrix(vec[..., 3:9]))
    return np.concatenate([vec[..., :3], q], axis=-1)


def _lookup(values, key):
    singular = key[:-1] if key.endswith(("_states", "_poses")) else key
    for candidate in (key, singular):
        for name in (candidate, f"state.{candidate}", f"state/{candidate}"):
            if name in values:
                return values[name]
    return None



def _field(values, key, width):
    value = _lookup(values, key)
    if value is None:
        raise KeyError(f"Missing MobileArena EE field {key}")
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] != width or not np.isfinite(arr).all():
        raise ValueError(f"Invalid {key}: expected finite (...,{width}), got {arr.shape}")
    return arr


def pack_state(obs):
    values = obs.get("state") if isinstance(obs.get("state"), Mapping) else obs
    left = pose_to_model(_field(values, "left_base_ee_poses", 7))
    right = pose_to_model(_field(values, "right_base_ee_poses", 7))
    return np.concatenate([
        left, _field(values, "left_ee_joint_states", 12),
        right, _field(values, "right_ee_joint_states", 12),
        _field(values, "root_linear_velocity", 3)[..., :2],
        _field(values, "root_angular_velocity", 3)[..., 2:3],
    ], axis=-1).astype(np.float32)


def unpack_actions(chunk):
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[-1] != RAW_DIM or not np.isfinite(chunk).all():
        raise ValueError(f"Expected finite (T,45) EE actions, got {chunk.shape}")
    left = model_to_pose(chunk[:, :9]).astype(np.float32)
    right = model_to_pose(chunk[:, 21:30]).astype(np.float32)
    for pose in (left, right):
        if not np.allclose(np.linalg.norm(pose[:, 3:], axis=-1), 1.0, atol=1e-4):
            raise ValueError("Model emitted a degenerate EE rotation")
    # The simulator consumes three-component twists. Only vx/vy/wz are
    # predicted; the other components are explicit zero control commands.
    return [{
        "left_base_ee_pose": left[t],
        "left_ee_joint_state": row[9:21].copy(),
        "right_base_ee_pose": right[t],
        "right_ee_joint_state": row[30:42].copy(),
        "root_linear_velocity": np.array([row[42], row[43], 0], dtype=np.float32),
        "root_angular_velocity": np.array([0, 0, row[44]], dtype=np.float32),
    } for t, row in enumerate(chunk)]


def prepare_camera(value, camera_name):
    """Mirror converter 224x224 AREA resize, then reader's LANCZOS tile resize."""
    import cv2
    from PIL import Image
    arr = np.asarray(value)
    if arr.shape[:2] != (224, 224):
        arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA)
    size = (320, 256) if camera_name == "cam_head" else (160, 128)
    return Image.fromarray(np.ascontiguousarray(arr)).resize(size, Image.Resampling.LANCZOS)
