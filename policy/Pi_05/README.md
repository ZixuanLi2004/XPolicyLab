# Pi_05 for MobileArena

**Contributor:** RoboDojo Team | **Original code:** https://github.com/Physical-Intelligence/openpi

This adapter includes MobileArena evaluation, HDF5 conversion and full-parameter training. `openpi/` is the vendored model implementation. It uses M92UW joint45 control: two arms (7 each), two hands (12 each), waist (2), legs (2), and base velocity `[vx, vy, wz]`. State and action dimensions must match the checkpoint; old 75D checkpoints are unsupported by this configuration.

Shared launch conventions are in the [XPolicyLab README](../../README.md).

## Installation

```bash
cd /personal/lzx/MobileArena/XPolicyLab/policy/Pi_05
bash install.sh
```

OpenPI uses `openpi/.venv`. Evaluation argument 9 accepts `uv` or another OpenPI project directory containing `.venv`; the adapter still imports the model code from this checkout. Training/conversion can reuse an interpreter with `OPENPI_PYTHON=/path/to/python`.

## Data Processing

```bash
MOBILE_ARENA_RAW_ROOT=/path/to/raw/MobileArena \
  bash process_data.sh MobileArena mobile_arena_full mobile_arena joint
# Optional final arguments: max_episodes and raw_source_root.
```

The source is a directory containing task HDF5 episodes. If no path is provided, the converter uses `<MobileArena>/data/MobileArena`. It writes a **LeRobot v3.0** image dataset to `data/mobile_arena_data`, then runs OpenPI normalization-statistics generation. The keys are `observation.state`, `action`, and `observation.images.cam_head/cam_left_wrist/cam_right_wrist`. The MobileArena converter differs from the generic exporter by selecting the 45 controllable dimensions and resizing RGB images to 224×224 using OpenCV linear interpolation. Image bytes are decoded only through the shared XPolicyLab decoder.

`MOBILE_ARENA_REPO_ID` changes the dataset ID; `OPENPI_HF_LEROBOT_HOME` changes the dataset parent directory. An existing dataset is preserved unless `MOBILE_ARENA_OVERWRITE=true` is set. Normalization statistics are generated under `openpi/assets/pi05_mobile_arena_full/<repo_id>/norm_stats.json`; they must describe the dataset used for training. For an already converted dataset, run:

```bash
cd openpi
HF_LEROBOT_HOME=../data .venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_mobile_arena_full --repo-id mobile_arena_data
```

## Training

```bash
bash train.sh MobileArena mobile_arena_full mobile_arena joint 0 0,1,2,3
```

Defaults match the source MobileArena recipe: `pi05_mobile_arena_full`, action horizon 50, global batch 128, 60,000 steps, mean/std normalization, full fine-tuning from `pi05_base`, and new 45D state/action projections. Pretrained base weights are downloaded by OpenPI when needed.

`OPENPI_LEROBOT_REPO_ID` must match the conversion's `MOBILE_ARENA_REPO_ID` (both default to `mobile_arena_data`). `OPENPI_BATCH_SIZE`, `OPENPI_FSDP_DEVICES`, `OPENPI_TRAIN_CONFIG_NAME` and `OPENPI_LOCAL_CACHE_ROOT` override the respective settings. Checkpoints go to `checkpoints/MobileArena-<ckpt_name>-mobile_arena-joint-<seed>/`. The inherited training entry uses `--overwrite`, so choose a new checkpoint name when retaining a previous run.

## Evaluation

```bash
bash eval.sh MobileArena place_fruit_basket mobile_arena_full mobile_arena joint 0 0 1 uv RoboDojo
```

The checkpoint argument may be a short run name, full run directory name, or absolute directory. `deploy.yml` selects `pi05_mobile_arena_full`, `repo_id: mobile_arena_data`, `mobile_arena: true` and checkpoint step 59999. Set `repo_id` to the training dataset's asset ID when using another dataset. The saved checkpoint must contain parameters and its own normalization assets.

The server accepts already-decoded RGB observations. The returned root linear/angular velocities have three elements; unmodeled `vz/wx/wy` are zero. Batch observations are supported. The policy environment and simulator environment are separate. `EVAL_ENV_TYPE=debug` checks transport using an installed policy and a real checkpoint; no model-free inference is provided.
