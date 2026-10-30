#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from collections.abc import Mapping
from pathlib import Path
import os
from typing import Any

import cv2
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as _transforms

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

# Runtime keys emitted by the 45D MobileArena head.  The two root-velocity keys
# carry the three learned base controls (vx, vy, wz); unmodeled components are
# zero-filled by miverse_layout.unpack to preserve their standard 3D shape.
_MOBILE_ARENA_OUTPUT_KEYS = (
    "left_arm_joint_state",
    "right_arm_joint_state",
    "left_ee_joint_state",
    "right_ee_joint_state",
    "waist_joint_state",
    "leg_joint_state",
    "root_linear_velocity",
    "root_angular_velocity",
)


def _load_mobile_arena_layout():
    from openpi import mobile_arena_layout

    return mobile_arena_layout


def _as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.mobile_arena_mode = _as_bool(model_cfg.get("mobile_arena", model_cfg.get("miverse", False)))
        self.mobile_arena = _load_mobile_arena_layout() if self.mobile_arena_mode else None
        self.robot_action_dim_info = (
            None
            if self.mobile_arena_mode
            else (
                get_robot_action_dim_info(model_cfg["env_cfg_type"])
                if model_cfg.get("env_cfg_type") is not None
                else None
            )
        )
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    def get_model(self, model_cfg: dict[str, Any]):
        default_config = "pi05_miverse_full" if self.mobile_arena_mode else "pi05_base_aloha_full_sim_arx-x5_seed_0"
        train_config_name = model_cfg.get("train_config_name", default_config)
        repo_id = model_cfg.get("repo_id", "miverse_data" if self.mobile_arena_mode else "1118")
        model_root = _resolve_pi05_model_root(model_cfg)

        config = _config.get_config(train_config_name)
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        repack_transforms = None
        if self.mobile_arena_mode:
            repack_transforms = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "images/cam_high",
                            "observation/left_wrist_image": "images/cam_left_wrist",
                            "observation/right_wrist_image": "images/cam_right_wrist",
                            "observation/state": "state",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
        return _policy_config.create_trained_policy(
            config, str(model_root), repack_transforms=repack_transforms, norm_stats=norm_stats
        )

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = []
        for obs in obs_list:
            encoded = encode_obs(obs, self.action_type, self.robot_action_dim_info, self.mobile_arena)
            if self.mobile_arena is not None and not encoded["prompt"]:
                encoded["prompt"] = self.task_name or ""
            encoded_obs_list.append(encoded)
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        env_to_batch = {env_idx: index for index, env_idx in enumerate(self._latest_env_idx_list)}
        batch_indices = [env_to_batch[env_idx] for env_idx in env_idx_list]
        action_list = []

        for batch_index in batch_indices:
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            actions = self.policy.infer(single_observation, **kwargs)["actions"]
            if self.mobile_arena is not None:
                action_list.append(unpack_mobile_arena_actions(actions, self.mobile_arena))
            elif self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()


def encode_obs(observation, action_type, robot_action_dim_info, mobile_arena=None):
    if mobile_arena is not None:
        return encode_mobile_arena_obs(observation, mobile_arena)

    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}



def _extract_mobile_arena_image(observation: dict[str, Any], names: tuple[str, ...]):
    source = observation.get("images") or observation.get("vision") or {}
    for name in names:
        if name not in source:
            continue
        image = source[name]
        if isinstance(image, Mapping):
            for key in ("color", "rgb", "image"):
                if key in image:
                    return image[key]
        else:
            return image
    raise KeyError(f"Could not find MiVerse camera; tried {names}")


def encode_mobile_arena_obs(observation: dict[str, Any], mobile_arena) -> dict[str, Any]:
    raw_state = observation.get("state")
    if isinstance(raw_state, Mapping):
        state = mobile_arena.pack(raw_state)
    else:
        state = np.asarray(raw_state, dtype=np.float32)
        if state.shape[-1] != mobile_arena.MODEL_DIM:
            state = mobile_arena.pack(observation)

    images = {
        "cam_high": ensure_chw_uint8(
            _extract_mobile_arena_image(observation, ("cam_head", "primary", "head_camera", "top_camera")),
            size=(224, 224),
        ),
        "cam_left_wrist": ensure_chw_uint8(
            _extract_mobile_arena_image(observation, ("cam_left_wrist", "wrist_left", "left_camera")),
            size=(224, 224),
        ),
        "cam_right_wrist": ensure_chw_uint8(
            _extract_mobile_arena_image(observation, ("cam_right_wrist", "wrist_right", "right_camera")),
            size=(224, 224),
        ),
    }
    prompt = observation.get("instruction") or observation.get("instructions") or ""
    if isinstance(prompt, (list, tuple)):
        prompt = prompt[0] if prompt else ""
    return {"state": np.asarray(state, dtype=np.float32), "images": images, "prompt": str(prompt)}


def unpack_mobile_arena_actions(actions, mobile_arena) -> list[dict[str, np.ndarray]]:
    """Return only keys represented by the learned 45D MobileArena head."""
    actions = np.asarray(actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    keyed = mobile_arena.unpack(actions, key_style="singular")
    # Keep deployment output closed over the trained contract.  In particular,
    # do not synthesize pose/head keys that are present in the raw HDF5 schema.
    missing = [key for key in _MOBILE_ARENA_OUTPUT_KEYS if key not in keyed]
    if missing:
        raise KeyError(f"MiVerse layout did not provide output keys: {missing}")
    return [
        {key: np.asarray(keyed[key])[step] for key in _MOBILE_ARENA_OUTPUT_KEYS}
        for step in range(actions.shape[0])
    ]


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image, size=None):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if size is not None:
        # Match process_mobile_arena.py before OpenPI's aspect-preserving resize.
        image_hwc = cv2.resize(image_hwc, size, interpolation=cv2.INTER_LINEAR)
        if image_hwc.ndim == 2:
            image_hwc = image_hwc[..., None]

    return np.transpose(image_hwc, (2, 0, 1))
