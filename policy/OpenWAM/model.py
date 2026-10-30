"""MobileArena M92UW EE45 adapter for the 80-D unified OpenWAM model.

Three RGB views and base-frame end-effector poses follow the MobileArena converter
and reader. Batched inference makes one generate_batch call for active envs.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.policy.OpenWAM import mobile_arena_ee_adapter

POLICY_DIR = Path(__file__).resolve().parent
# Vendored OpenWAM source tree (weights excluded).
VENDORED_OPENWAM_ROOT = POLICY_DIR / "OpenWAM"
CHECKPOINTS_DIR = POLICY_DIR / "checkpoints"

# Fixed obs -> OpenWAM payload camera mapping (matches training camera_layout).
_CAMERA_TO_PAYLOAD = {
    "cam_head": "head_camera",
    "cam_left_wrist": "left_wrist_camera",
    "cam_right_wrist": "right_wrist_camera",
}


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _resolve_ckpt_dir(model_cfg: dict) -> Path:
    """Resolve the OpenWAM checkpoint directory via the shared XPolicyLab rules.

    Precedence: explicit ``ckpt_dir`` / path-like ``ckpt_name`` >
    ``checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/`` >
    ``checkpoints/<ckpt_name>/``. The directory must contain ``config.yaml``.
    """
    root = resolve_checkpoint_root(
        model_cfg,
        CHECKPOINTS_DIR,
        policy_dir=POLICY_DIR,
        explicit_keys=("ckpt_dir", "checkpoint_path", "ckpt_path", "model_dir"),
        must_exist=True,
    )
    if not (root / "config.yaml").is_file():
        raise FileNotFoundError(f"checkpoint dir has no config.yaml: {root}")
    return root


def _resolve_openwam_root(openwam_root: Any) -> Path:
    """Resolve the OpenWAM source root and make it win the import resolution.

    Default is the vendored copy next to this file. The root is prepended to
    sys.path unconditionally: the policy conda env typically carries an
    editable install pointing at the external dev repo, and the vendored /
    explicitly-configured tree must take precedence over it.
    """
    root = VENDORED_OPENWAM_ROOT if _is_none_like(openwam_root) else Path(str(openwam_root)).expanduser()
    if not (root / "openwam" / "__init__.py").is_file():
        raise FileNotFoundError(f"openwam package not found under openwam_root: {root}")
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    if "openwam" in sys.modules:
        loaded = Path(getattr(sys.modules["openwam"], "__file__", "") or "").resolve()
        if not str(loaded).startswith(str(root.resolve())):
            raise RuntimeError(
                f"openwam was already imported from {loaded}, not from openwam_root {root}; "
                "start the server in a fresh process."
            )
    return root


def _decode_instruction(value: Any, fallback: str) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else (value.reshape(-1)[0] if value.size else None)
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _to_pil(value: Any, ctx: str):
    """Standardize a decoded obs camera frame into an RGB PIL image."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value if value.mode == "RGB" else value.convert("RGB")
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[2] < 3 or 0 in arr.shape:
        raise ValueError(f"{ctx}: expected decoded (H, W, >=3) image array, got shape {arr.shape}")
    arr = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(arr)) <= 1.5 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr), mode="RGB")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.raw_action_dim = mobile_arena_ee_adapter.RAW_DIM
        action_type = self.model_cfg.get("action_type") or "ee"
        if action_type != "ee":
            raise ValueError(f"MobileArena OpenWAM requires action_type='ee', got {action_type!r}")
        configured_dim = int(self.model_cfg.get("action_dim") or self.raw_action_dim)
        if configured_dim != self.raw_action_dim:
            raise ValueError(f"MobileArena OpenWAM expects action_dim=45, got {configured_dim}")
        self.openwam_root = _resolve_openwam_root(self.model_cfg.get("openwam_root"))

        self.default_instruction = str(self.model_cfg.get("default_instruction") or "follow the instruction")
        replan = self.model_cfg.get("replan_steps")
        self.replan_steps = None if _is_none_like(replan) else int(replan)
        if self.replan_steps is not None and self.replan_steps <= 0:
            raise ValueError(f"replan_steps must be positive or null, got {self.replan_steps}")

        # Per-env stored observations (keyed by env_idx, order preserved).
        self._batch: dict[int, dict] = {}
        self._order: list[int] = []

        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))
        self._engine = None
        self._wam_policy = None
        self._preprocessor = None
        if self.allow_dummy_policy:
            print("[OpenWAM] allow_dummy_policy=true; checkpoint loading skipped (protocol debug only).")
            return

        ckpt_dir = str(_resolve_ckpt_dir(self.model_cfg))
        from omegaconf import OmegaConf
        saved_cfg = OmegaConf.load(Path(ckpt_dir) / "config.yaml")
        saved_type = OmegaConf.select(saved_cfg, "dataloader.type")
        if saved_type != "mobile_arena_ee":
            raise ValueError(f"MobileArena OpenWAM requires dataloader=mobile_arena_ee, got {saved_type!r}")
        if OmegaConf.select(saved_cfg, "model.architecture.action_dim") != 80:
            raise ValueError("MobileArena EE checkpoint must use the original 80D OpenWAM model")
        if OmegaConf.select(saved_cfg, "dataloader.action_mode") != "mobile_arena_ee":
            raise ValueError("MobileArena EE checkpoint requires dataloader.action_mode=mobile_arena_ee")
        stats = np.load(Path(ckpt_dir) / "normalization_stats.npy", allow_pickle=True).item()
        if "mobile_arena_ee_state" not in stats:
            raise ValueError("MobileArena EE checkpoint is missing its separate state normalization statistics")

        # OpenWAM's own deploy defaults; the correctness-critical keys are
        # force-overridden below, so no separate pinned eval yaml is needed.
        deploy_config = self.model_cfg.get("openwam_deploy_config")
        if _is_none_like(deploy_config):
            deploy_config = str(self.openwam_root / "configs" / "deploy.yaml")
        device = str(self.model_cfg.get("device") or "cuda")

        from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

        deploy_cfg = _load_deploy_yaml(deploy_config)
        # Correctness-first: batch inference forbids the single-stream /
        # shape-sensitive acceleration paths. Force them off even if the yaml
        # drifts; engine.generate_batch fails fast if these were re-enabled.
        OmegaConf.update(deploy_cfg, "optimization.dit_cache.enabled", False, merge=False)
        OmegaConf.update(deploy_cfg, "optimization.compile.enabled", False, merge=False)
        OmegaConf.update(deploy_cfg, "optimization.decode_video", False, merge=False)
        OmegaConf.update(deploy_cfg, "inference.inference_mode", "sync", merge=False)

        print(f"[OpenWAM] loading checkpoint from {ckpt_dir} on {device} ...")
        server = build_server_from_config(deploy_cfg, ckpt_dir, device=device, ckpt_name=self.model_cfg.get("checkpoint_file") or None)
        server._init_policy()
        self._engine = server.engine
        self._wam_policy = server._policy  # for the binary-dim legality projection
        self._preprocessor = server._obs_preprocessor

        if self.replan_steps is None:
            horizon = OmegaConf.select(server.cfg, "inference.inference_horizon", default=None)
            self.replan_steps = None if horizon is None else int(horizon)

        resolved = {
            "denoise_steps": OmegaConf.select(server.cfg, "inference.denoise_steps"),
            "denoise_mode": OmegaConf.select(server.cfg, "inference.denoise_mode"),
            "num_frames": OmegaConf.select(server.cfg, "inference.num_frames"),
            "video_num_frames": OmegaConf.select(server.cfg, "inference.video_num_frames"),
            "replan_steps": self.replan_steps or "full chunk",
            "dit_cache": OmegaConf.select(server.cfg, "optimization.dit_cache.enabled"),
            "compile": OmegaConf.select(server.cfg, "optimization.compile.enabled"),
        }
        print(f"[OpenWAM] initialized | {resolved}")

    def _encode_obs(self, obs: Mapping) -> dict:
        vision = obs.get("vision")
        if not isinstance(vision, Mapping):
            raise ValueError("obs['vision'] must be a mapping of camera name -> {'color': image}")
        images = {}
        for camera_name, payload_name in _CAMERA_TO_PAYLOAD.items():
            camera = vision.get(camera_name)
            if not isinstance(camera, Mapping) or "color" not in camera:
                raise KeyError(f"missing required camera vision/{camera_name}/color")
            images[payload_name] = _to_pil(camera["color"], ctx=f"vision/{camera_name}/color")
            images[payload_name] = mobile_arena_ee_adapter.prepare_camera(images[payload_name], camera_name)

        state = obs.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("obs['state'] must be a mapping")
        model_state = mobile_arena_ee_adapter.pack_state(obs)

        instruction = _decode_instruction(
            obs.get("instruction", obs.get("instructions", obs.get("task_instruction"))), self.default_instruction
        )
        return {
            "images": images,
            "prompt": instruction,
            "state": np.asarray(model_state, dtype=np.float32).tolist(),
        }

    def _conditions(self, payload: dict) -> dict:
        """OpenWAM server-equivalent preprocessing: compose multiview + wrap."""
        obs = self._preprocessor.preprocess(dict(payload))
        cond = {
            "observation": obs,
            "first_frame_image": [obs["image"]],
            "prompt": obs["prompt"],
        }
        if obs.get("state") is not None:
            cond["proprio"] = obs["state"]
        return cond

    def _hold_position_chunk(self, payload: dict) -> list[dict]:
        """Dummy-mode chunk for checking transport without loading weights."""
        steps = self.replan_steps or 8
        model_state = np.asarray(payload["state"], dtype=np.float64)
        return mobile_arena_ee_adapter.unpack_actions(np.tile(model_state, (steps, 1)))

    # ------------------------------------------------------------------
    # XPolicyLab protocol
    # ------------------------------------------------------------------
    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, Mapping):
            obs_list = [obs_list]
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        self._batch = {}
        self._order = []
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", index))
            self._batch[env_idx] = self._encode_obs(obs)
            self._order.append(env_idx)

    def get_action(self):
        env_idx = self._order[0] if self._order else 0
        return self.get_action_batch([env_idx])[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(self._order)
        elif isinstance(env_idx_list, np.ndarray):
            env_idx_list = env_idx_list.reshape(-1).tolist()
        elif isinstance(env_idx_list, (int, np.integer)):
            env_idx_list = [int(env_idx_list)]
        else:
            env_idx_list = list(env_idx_list)
        env_idx_list = [int(e) for e in env_idx_list]
        if not env_idx_list:
            raise ValueError("get_action_batch received an empty env_idx_list.")

        missing = [e for e in env_idx_list if e not in self._batch]
        if missing:
            raise ValueError(f"No stored observation for env_idx {missing}; call update_obs_batch first.")
        payloads = [self._batch[e] for e in env_idx_list]

        if self.allow_dummy_policy:
            return [self._hold_position_chunk(payload) for payload in payloads]

        conditions = [self._conditions(p) for p in payloads]
        result = self._engine.generate_batch(conditions)
        actions = result["actions"]
        if hasattr(actions, "cpu"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        expected_dim = self.raw_action_dim
        if actions.ndim != 3 or actions.shape[0] != len(env_idx_list) or actions.shape[2] != expected_dim:
            raise RuntimeError(
                f"generate_batch returned actions of shape {actions.shape}; "
                f"expected (B={len(env_idx_list)}, T, {expected_dim})."
            )
        # Same final legality projection the OpenWAM server applies (no-op when
        # the checkpoint declares no binary command dims).
        actions = self._wam_policy._project_binary_dims(actions)

        n_exec = actions.shape[1]
        if self.replan_steps is not None:
            n_exec = min(self.replan_steps, n_exec)
        return [mobile_arena_ee_adapter.unpack_actions(actions[b, :n_exec]) for b in range(actions.shape[0])]

    def reset(self):
        self._batch = {}
        self._order = []
