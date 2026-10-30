"""MobileArena m75 <-> Xiaomi XR1 60D adapter.

The native XR1 layout is (30, 60).  MobileArena EE poses are converted from
6D rotation to axis-angle in the native EE slots.  The MobileArena action policy
uses only four explicit active ranges:
  [0:6]   left EE position + axis-angle
  [8:14]  right EE position + axis-angle
  [17:20] native base velocity [vx, vy, wz]
  [20:44] left/right 12D dexterous hands
All other native action slots are masked and emitted as zero.  EE and hand
values are relative deltas; base velocity is an absolute target velocity.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

MOBILE_ARENA_DIM = 75
XIAOMI_DIM = 60
# Explicit action mask.  The 60D native tensor still contains reserved
# fields, but only these 39 values are trained and sent to the robot.
ACTION_RANGES = ((0, 6), (8, 14), (17, 20), (20, 44))
ACTIVE_DIM = sum(hi - lo for lo, hi in ACTION_RANGES)
_EPS = 1e-8


def _rot6d_to_matrix(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    a1, a2 = v[..., :3], v[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), _EPS, None)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / np.clip(np.linalg.norm(a2, axis=-1, keepdims=True), _EPS, None)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def _matrix_to_rot6d(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return np.concatenate((m[..., :, 0], m[..., :, 1]), axis=-1)


def _matrix_to_axis_angle(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    skew = np.stack((m[..., 2, 1] - m[..., 1, 2],
                     m[..., 0, 2] - m[..., 2, 0],
                     m[..., 1, 0] - m[..., 0, 1]), axis=-1)
    sn = np.linalg.norm(skew, axis=-1)
    angle = np.arctan2(sn, trace - 1.0)
    out = skew * (angle / np.clip(sn, _EPS, None))[..., None]
    deg = sn < 1e-7
    if np.any(deg):
        zero = deg & (trace - 1.0 > 0.0)
        out = np.where(zero[..., None], skew * 0.5, out)
        pi = deg & ~zero
        if np.any(pi):
            # Stable enough for the rarely occurring exact-pi case.  The
            # largest diagonal of (R+I)/2 identifies the rotation axis.
            sm = (m[pi] + np.eye(3)) * 0.5
            k = np.argmax(np.diagonal(sm, axis1=-2, axis2=-1), axis=-1)
            axis = np.stack((sm[np.arange(len(k)), :, 0],
                             sm[np.arange(len(k)), :, 1],
                             sm[np.arange(len(k)), :, 2]), axis=-1)
            axis = axis[np.arange(len(k)), k]
            axis = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), _EPS, None)
            sign = np.sign(np.sum(axis * skew[pi], axis=-1))
            sign = np.where(sign == 0.0, 1.0, sign)
            out[pi] = axis * (angle[pi] * sign)[..., None]
    return out


def _axis_angle_to_matrix(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = v / np.clip(angle, _EPS, None)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    k = np.stack((
        np.stack((zero, -z, y), axis=-1),
        np.stack((z, zero, -x), axis=-1),
        np.stack((-y, x, zero), axis=-1),
    ), axis=-2)
    eye = np.broadcast_to(np.eye(3), k.shape).copy()
    return eye + np.sin(angle[..., None]) * k + (1.0 - np.cos(angle[..., None])) * (k @ k)


def _rot6d_to_aa(v: np.ndarray) -> np.ndarray:
    return _matrix_to_axis_angle(_rot6d_to_matrix(v))


def _get(mapping: Mapping[str, object], *names: str, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
        for prefix in ("state.", "state/"):
            if prefix + name in mapping:
                return mapping[prefix + name]
    return default


_RAW_FIELDS = (
    ("left_arm_joint_states", 7),
    ("right_arm_joint_states", 7),
    ("left_ee_joint_states", 12),
    ("right_ee_joint_states", 12),
    ("left_base_ee_poses", 7),
    ("right_base_ee_poses", 7),
    ("waist_joint_states", 2),
    ("leg_joint_states", 2),
    ("head_base_poses", 7),
    ("root_linear_velocity", 3),
    ("root_angular_velocity", 3),
)


def raw_fields_to_mobile_arena75(values: Mapping[str, object]) -> np.ndarray:
    """Pack plural MobileArena HDF5/JSON fields into vectorized model-75 rows."""
    parts = []
    lead = None
    for name, dim in _RAW_FIELDS:
        value = values.get(name)
        if value is None:
            # Accept the singular spelling used by XPolicyLab runtime messages.
            value = values.get(name[:-1] if name.endswith("s") else name)
        if value is None:
            raise KeyError(name)
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape[-1] != dim:
            raise ValueError(f"{name} expected last dim {dim}, got {arr.shape}")
        if lead is None:
            lead = arr.shape[:-1]
        if arr.shape[:-1] != lead:
            raise ValueError(f"{name} lead shape {arr.shape[:-1]} != {lead}")
        parts.append(arr)
    return raw_to_mobile_arena75(np.concatenate(parts, axis=-1))

def runtime_state_to_mobile_arena75(state: Mapping[str, object], allow_missing: bool = True) -> np.ndarray:
    """Pack XPolicyLab's 11 runtime state keys into MobileArena m75."""
    def vec(*names, dim):
        value = _get(state, *names)
        if value is None:
            if not allow_missing:
                raise KeyError(names[0])
            return np.zeros(dim, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if value.size != dim:
            raise ValueError(f"{names[0]} expected {dim}, got {value.shape}")
        return value

    def pose(*names):
        value = _get(state, *names)
        if value is None:
            if not allow_missing:
                raise KeyError(names[0])
            return np.array([0., 0., 0., 1., 0., 0., 0.], dtype=np.float64)
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if value.size != 7:
            raise ValueError(f"{names[0]} expected 7, got {value.shape}")
        return value

    def base_pose(name, plural_name):
        # MobileArena is trained on base_link EE poses.  Never fall back to the
        # world-frame left_ee_pose/right_ee_pose fields: silently mixing the
        # two frames corrupts both the state encoder and the relative-action
        # reconstruction.
        value = _get(state, name, plural_name)
        if value is None:
            raise KeyError(name)
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if value.size != 7:
            raise ValueError(f"{name} expected 7, got {value.shape}")
        return value

    raw = np.concatenate((
        vec("left_arm_joint_state", "left_arm_joint_states", dim=7),
        vec("right_arm_joint_state", "right_arm_joint_states", dim=7),
        vec("left_ee_joint_state", "left_ee_joint_states", dim=12),
        vec("right_ee_joint_state", "right_ee_joint_states", dim=12),
        base_pose("left_base_ee_pose", "left_base_ee_poses"),
        base_pose("right_base_ee_pose", "right_base_ee_poses"),
        vec("waist_joint_state", "waist_joint_states", dim=2),
        vec("leg_joint_state", "leg_joint_states", dim=2),
        pose("head_base_pose", "head_base_poses", "base_head_pose"),
        vec("root_linear_velocity", dim=3),
        vec("root_angular_velocity", dim=3),
    ))
    return raw_to_mobile_arena75(raw)


def raw_to_mobile_arena75(raw: np.ndarray) -> np.ndarray:
    """Convert raw 69D key-concatenation with quaternion poses to m75."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[-1] != 69:
        raise ValueError(f"expected raw dim 69, got {raw.shape}")
    out = np.zeros(raw.shape[:-1] + (75,), dtype=np.float64)
    out[..., :38] = raw[..., :38]
    out[..., 38:47] = np.concatenate((raw[..., 38:41], _matrix_to_rot6d(_quat_wxyz_to_matrix(raw[..., 41:45]))), axis=-1)
    out[..., 47:56] = np.concatenate((raw[..., 45:48], _matrix_to_rot6d(_quat_wxyz_to_matrix(raw[..., 48:52]))), axis=-1)
    out[..., 56:60] = raw[..., 52:56]
    out[..., 60:69] = np.concatenate((raw[..., 56:59], _matrix_to_rot6d(_quat_wxyz_to_matrix(raw[..., 59:63]))), axis=-1)
    out[..., 69:75] = raw[..., 63:69]
    return out


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), _EPS, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((
        np.stack((1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)), axis=-1),
        np.stack((2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)), axis=-1),
        np.stack((2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)), axis=-1),
    ), axis=-2)


def _matrix_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    # scipy is already a dependency of the XR1 environment; use it only for
    # deployment conversion where this path is small and infrequent.
    from scipy.spatial.transform import Rotation
    q_xyzw = Rotation.from_matrix(np.asarray(m, dtype=np.float64)).as_quat()
    q = q_xyzw[..., [3, 0, 1, 2]]
    return np.where(q[..., :1] < 0.0, -q, q)


def state75_to_xiaomi60(state75: np.ndarray) -> np.ndarray:
    """Pack an m75 state into the 60D XR1 state tensor.

    Layout (arm joints keep the released checkpoint's channel meaning; the
    dexterous hands live in the region the checkpoint never trained):

        [0:7]    left arm joint angles    <- pretrained channel, inherited
        [7]      zero   (this robot has no gripper)
        [8:15]   right arm joint angles   <- pretrained channel, inherited
        [15]     zero   (this robot has no gripper)
        [16:28]  left hand, 12 joints     <- new, projector learns from scratch
        [28:40]  right hand, 12 joints    <- new, projector learns from scratch
        [40:60]  zero padding

    mibot.utils.io.compose_state puts a single gripper scalar at [7] and [15].
    The MobileArena robot (m92uw) has a 12-DoF hand instead of a gripper, so those
    two channels hold a constant 0: zero is a valid gripper position in the
    pretrained distribution, and no non-gripper quantity is pushed into a
    channel the checkpoint trained with gripper semantics.

    The hands get their own 24 channels.  Those input columns are dead in the
    released state_projector (measured column norm ~0.136 versus 0.20-0.25 for
    the trained channels), so they are learned during fine-tuning while the
    arm joint channels and the rest of the backbone are inherited.
    """
    x = np.asarray(state75, dtype=np.float64)
    if x.shape[-1] != MOBILE_ARENA_DIM:
        raise ValueError(f"expected m75, got {x.shape}")
    out = np.zeros(x.shape[:-1] + (XIAOMI_DIM,), dtype=np.float64)
    out[..., 0:7] = x[..., 0:7]        # left arm joints  (pretrained channel)
    out[..., 8:15] = x[..., 7:14]      # right arm joints (pretrained channel)
    out[..., 16:28] = x[..., 14:26]    # left hand, 12 joints
    out[..., 28:40] = x[..., 26:38]    # right hand, 12 joints
    return out


def state75_to_xiaomi60_action(current75: np.ndarray, target75: np.ndarray) -> np.ndarray:
    """Encode absolute MobileArena target(s) as masked XR1 actions.

    EE, dexterous-hand and orientation values are relative to the current
    state.  Native base velocity is an absolute [vx, vy, wz] target.
    """
    cur = np.asarray(current75, dtype=np.float64)
    tar = np.asarray(target75, dtype=np.float64)
    if cur.shape[-1] != MOBILE_ARENA_DIM or tar.shape[-1] != MOBILE_ARENA_DIM:
        raise ValueError("current and target must end in 75")
    out = np.zeros(np.broadcast_shapes(cur.shape[:-1], tar.shape[:-1]) + (XIAOMI_DIM,), dtype=np.float64)
    lc = _rot6d_to_matrix(cur[..., 41:47])
    lt = _rot6d_to_matrix(tar[..., 41:47])
    rc = _rot6d_to_matrix(cur[..., 50:56])
    rt = _rot6d_to_matrix(tar[..., 50:56])
    out[..., 0:3] = np.einsum("...ji,...j->...i", lc, tar[..., 38:41] - cur[..., 38:41])
    out[..., 3:6] = _matrix_to_axis_angle(np.einsum("...ji,...jk->...ik", lc, lt))
    out[..., 8:11] = np.einsum("...ji,...j->...i", rc, tar[..., 47:50] - cur[..., 47:50])
    out[..., 11:14] = _matrix_to_axis_angle(np.einsum("...ji,...jk->...ik", rc, rt))
    out[..., 17:19] = tar[..., 69:71]  # absolute vx, vy
    out[..., 19] = tar[..., 74]         # absolute wz
    out[..., 20:32] = tar[..., 14:26] - cur[..., 14:26]
    out[..., 32:44] = tar[..., 26:38] - cur[..., 26:38]
    return out


def xiaomi60_action_to_mobile_arena75(action60: np.ndarray, current75: np.ndarray) -> np.ndarray:
    """Decode masked XR1 actions to absolute m75 targets for deployment."""
    a = np.asarray(action60, dtype=np.float64)
    cur = np.asarray(current75, dtype=np.float64)
    if a.shape[-1] != XIAOMI_DIM or cur.shape[-1] != MOBILE_ARENA_DIM:
        raise ValueError("action must be 60D and current state 75D")
    if cur.ndim == a.ndim - 1:
        cur = np.broadcast_to(cur, a.shape[:-1] + (MOBILE_ARENA_DIM,))
    out = np.array(cur, copy=True)
    lc = _rot6d_to_matrix(cur[..., 41:47])
    rc = _rot6d_to_matrix(cur[..., 50:56])
    out[..., 38:41] = cur[..., 38:41] + np.einsum("...ij,...j->...i", lc, a[..., 0:3])
    out[..., 41:47] = _matrix_to_rot6d(np.einsum("...ij,...jk->...ik", lc, _axis_angle_to_matrix(a[..., 3:6])))
    out[..., 47:50] = cur[..., 47:50] + np.einsum("...ij,...j->...i", rc, a[..., 8:11])
    out[..., 50:56] = _matrix_to_rot6d(np.einsum("...ij,...jk->...ik", rc, _axis_angle_to_matrix(a[..., 11:14])))
    out[..., 69:71] = a[..., 17:19]
    out[..., 74] = a[..., 19]
    out[..., 14:26] = cur[..., 14:26] + a[..., 20:32]
    out[..., 26:38] = cur[..., 26:38] + a[..., 32:44]
    # All masked native slots remain equal to the current observation.
    return out


def unpack_mobile_arena75(vec: np.ndarray) -> dict[str, np.ndarray]:
    """Return MobileArena runtime keys with EE poses in base_link coordinates.

    Arm joints, waist, legs and head are not action outputs of this adapter.
    The two velocity keys retain their 3D MobileArena shape; only vx/vy and wz
    are changed by the inverse action mapping.
    """
    x = np.asarray(vec, dtype=np.float64)
    if x.shape[-1] != MOBILE_ARENA_DIM:
        raise ValueError(f"expected m75, got {x.shape}")
    def pose(lo: int) -> np.ndarray:
        q = _matrix_to_quat_wxyz(_rot6d_to_matrix(x[..., lo + 3:lo + 9]))
        return np.concatenate((x[..., lo:lo + 3], q), axis=-1).astype(np.float32)
    return {
        "left_base_ee_pose": pose(38),
        "right_base_ee_pose": pose(47),
        "left_ee_joint_state": x[..., 14:26].astype(np.float32),
        "right_ee_joint_state": x[..., 26:38].astype(np.float32),
        "root_linear_velocity": x[..., 69:72].astype(np.float32),
        "root_angular_velocity": x[..., 72:75].astype(np.float32),
    }

def active_action_ranges() -> list[list[int]]:
    return [[lo, hi] for lo, hi in ACTION_RANGES]
