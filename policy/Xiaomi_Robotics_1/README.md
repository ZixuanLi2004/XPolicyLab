# Xiaomi_Robotics_1 for MobileArena

**Original code:** [Xiaomi-Robotics-1](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1)

This adapter vendors `xiaomi_robotics_1/xr1` and supports MobileArena HDF5 conversion,
training, and batched WebSocket evaluation. It uses the source MobileArena
`mobile_arena_official_state` recipe: 60-dimensional state, 39 active action dimensions,
30-step chunks, and M92UW base-frame end-effector actions (`action_type: ee`).
Shared deployment conventions are described in the [root README](../../README.md).

## Installation

```bash
cd policy/Xiaomi_Robotics_1
MIBOT_CONDA_ENV=mibot bash install.sh
conda activate mibot
```

The policy environment is separate from the simulator environment. The source
recipe requires a CUDA setup suitable for PyTorch 2.8/cu128 and FlashAttention;
see [INSTALLATION.md](INSTALLATION.md). No weights are copied by the merge.

## Data Processing

This policy uses native XR1 JSON trajectories and RGB MP4s, not a LeRobot export.
The converter consumes original MobileArena HDF5 `state`, `action`, and the three
head/wrist cameras. It obtains training targets from `action/*`; it does not
substitute next-frame state. Images are decoded through the shared
`XPolicyLab.utils.process_data.decode_image_bit`, preserving both legacy and
marked RGB buffers.

```bash
conda activate mibot
MOBILE_ARENA_RAW_ROOT=/path/to/MobileArena/hdf5 \
  bash process_data.sh --workers 16
```

Default output: `data/mobile_arena_native`; config:
`xiaomi_robotics_1/xr1/configs/data/mobile_arena_native.yaml`.
`MOBILE_ARENA_OUTPUT_ROOT` selects another output directory. `process_mobile_arena.sh` is an
alias. `--max-episodes 2` limits a trial conversion; `--stats-only` recomputes
statistics from HDF5 without rewriting videos. Always recompute stats for your
training data; the included config retains the source recipe's statistics.

The converter stores arm joints in state slots `0:7, 8:15`, 12-joint hands in
`16:28, 28:40`, and zeros elsewhere. Targets use EE deltas in `0:6, 8:14`, absolute
`vx, vy, wz` in `17:20`, and hand deltas in `20:44`. It writes per-horizon action
mean/std and state quantiles together with the 39-dimension action mask.

## Training

Run from the activated `mibot` environment:

```bash
PRETRAINED_PATH=/path/to/pretrained/model_states.pt \
  bash train.sh MobileArena all mobile_arena ee 0 0,1,2,3
```

The first six arguments follow XPolicyLab's naming convention; further arguments
are Hydra overrides. Defaults follow the current source recipe: batch size 8 per
GPU, 60,000 steps, saves every 10,000 steps, peak LR `1e-5`, async training, and
frequency loss excluding base-velocity slots `17:20`. Override with `BATCH_SIZE`,
`MAX_STEPS`, `SAVE_INTERVAL`, `MAX_LR`, `NUM_WORKERS`, `RUN_ROOT`, or explicit
Hydra arguments. `MOBILE_ARENA_OUTPUT_ROOT` must match conversion.
For a separate dataset config, write it under `xr1/configs/data/` with
`process_data.sh --config ...`, then select its basename through `DATA_CONFIG_NAME`.

Multi-node training uses the same entry: set `WORLD_SIZE` to the **number of
nodes**, `RANK` to the node index, and a shared `MASTER_ADDR`/`MASTER_PORT` on each
node. For the source four-node/eight-GPU recipe, use `WORLD_SIZE=4` and GPU list
`0,1,2,3,4,5,6,7` on every node. No hostnames or private server paths are embedded.

Artifacts are written under
`train_runs/<bench>-<ckpt>-<env>-<action>-<seed>/project_xr1/<same-run-name>/`.
`checkpoints/<same-run-name>` points to that artifact directory. It contains
`config.py` and `last.ckpt/checkpoint/mp_rank_00_model_states.pt`.

## Evaluation

```bash
XR1_CKPT_DIR=/path/to/current_mobile_arena_checkpoint \
  bash eval.sh MobileArena place_kettle_on_tv_stand MobileArena mobile_arena ee 0 0 1 mibot RoboDojo
```

Use an actual task name from the selected MobileArena client. The normal
`setup_eval_policy_server.sh` / `setup_eval_env_client.sh` interfaces also support
external orchestration. Configure `model_dir` in `deploy.yml`, set
`XR1_CKPT_DIR`, or provide `ckpt_name` as a checkpoint path. Otherwise shared
checkpoint resolution finds `checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>`.
`vlm_processor_path` may point to a local Qwen3-VL processor cache.

Runtime requires all 11 MobileArena state fields, including `left_base_ee_pose` and
`right_base_ee_pose`; world-frame EE poses are not substitutes. Each predicted
step is restored relative to the same observed state. Only `vx, vy, wz` change;
the other three velocity components retain the observation values. Old
`mobile_arena_formal`/`mobile_arena_restart` checkpoint layouts are not supported.

## Validation

Static checks and CPU conversion/layout checks can run without loading a model.
Actual training, model loading and simulator scores require the corresponding
weights and GPU environment; those are separate runtime acceptance checks.
