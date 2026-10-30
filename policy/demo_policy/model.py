import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info


class Model(ModelTemplate):
    """Weight-free adapter that holds the observed posture for interface checks."""

    def __init__(self, model_cfg):
        self.action_type = model_cfg["action_type"]
        if self.action_type not in ("joint", "ee"):
            raise ValueError(f"Unsupported action type: {self.action_type}")
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        num_arms = len(self.robot_action_dim_info["arm_dim"])
        if num_arms != len(self.robot_action_dim_info["ee_dim"]):
            raise ValueError("Arm and end-effector counts must match")
        if num_arms not in (1, 2):
            raise NotImplementedError(f"Unsupported number of arms: {num_arms}")
        self.arm_prefixes = [""] if num_arms == 1 else ["left_", "right_"]
        self.reset()

    def update_obs(self, obs):
        self.obs = obs

    def update_obs_batch(self, obs_list):
        self.obs_batch = {
            obs.get("additional_info", {}).get("env_idx", obs.get("env_idx", index)): obs
            for index, obs in enumerate(obs_list)
        }

    @staticmethod
    def _state_vector(state, key, dim):
        value = np.asarray(state[key], dtype=np.float32)
        if value.shape != (dim,):
            raise ValueError(f"{key}: expected shape {(dim,)}, got {value.shape}")
        return value.copy()

    def _action_chunk(self, obs):
        state = obs["state"]
        action = {}
        for index, prefix in enumerate(self.arm_prefixes):
            if self.action_type == "joint":
                key = f"{prefix}arm_joint_state"
                dim = self.robot_action_dim_info["arm_dim"][index]
            else:
                # MobileArena exposes poses in the robot base frame by default.
                key = f"{prefix}base_ee_pose"
                if key not in state:
                    key = f"{prefix}ee_pose"
                dim = 7
            action[key] = self._state_vector(state, key, dim)
            hand_key = f"{prefix}ee_joint_state"
            action[hand_key] = self._state_vector(
                state, hand_key, self.robot_action_dim_info["ee_dim"][index]
            )

        if self.action_type == "joint":
            for part in ("waist", "leg"):
                key = f"{part}_joint_state"
                if key in state:
                    dim = sum(self.robot_action_dim_info[f"{part}_dim"])
                    action[key] = self._state_vector(state, key, dim)
        elif "head_base_pose" in state:
            # In EE mode the whole-body IK owns waist and leg joints.
            action["head_base_pose"] = self._state_vector(state, "head_base_pose", 7)

        for key in ("root_linear_velocity", "root_angular_velocity"):
            if key in state:
                action[key] = np.zeros_like(self._state_vector(state, key, 3))

        return [{key: value.copy() for key, value in action.items()} for _ in range(2)]

    def get_action(self):
        if self.obs is None:
            raise RuntimeError("Call update_obs() before get_action()")
        return self._action_chunk(self.obs)

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = list(self.obs_batch)
        return [self._action_chunk(self.obs_batch[index]) for index in env_idx_list]

    def reset(self):
        self.obs = None
        self.obs_batch = {}
