# OpenWAM for MobileArena

**Contributor:** OpenWAM Contributors | **Paper:** An Open, Modular Exploration Towards Systematic World–Action Model Pretraining | **arXiv:** [2609.07398](https://arxiv.org/abs/2609.07398) | **Original code:** https://github.com/OpenWAM-Official/OpenWAM

This adapter includes MobileArena M92UW data conversion, fine-tuning and batched EE evaluation. The vendored implementation is in `OpenWAM/`. Runtime and training use 45 raw dimensions, scattered into the foundation model's original 80-dimensional action space. The obsolete 75-dimensional checkpoint layout is not supported.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use the OpenWAM policy environment with its CUDA-compatible PyTorch installed, then install this checkout and its vendored model:

```bash
conda activate openwam
cd XPolicyLab/policy/OpenWAM
bash install.sh
```

`install.sh` installs the dependencies declared by `OpenWAM/pyproject.toml`, including the PyAV and PyArrow conversion dependencies. Model weights and datasets are external resources; this repository does not copy them.

## Data Processing

The converter reads `<raw_root>/<task>/mobile_arena/data/episode_*.hdf5` and writes a **custom LeRobot v3 bucket**. Its image keys are `observation.images.cam_head`, `observation.images.cam_left_wrist`, and `observation.images.cam_right_wrist`; `observation.state` and `action` have 45 dimensions. This differs from the shared general-purpose LeRobot converters and must be produced with this policy's converter:

```bash
bash process_data.sh <raw_root> [output_directory] [converter_flags...]

# Example; activate the OpenWAM environment first.
bash process_data.sh /path/to/MobileArena/raw ./data/mobile_arena_ee --workers 16
```

The output defaults to `OPENWAM_DATASET_DIR` or `./data/mobile_arena_ee`. `OPENWAM_PYTHON` can select another Python executable. `--max-episodes N` limits a conversion; `--metadata-only` rebuilds metadata; completed episodes are skipped unless `--force` is passed. Use a separate output directory when changing the selected source episodes or image settings.

The layout is `left EE xyz+rot6d (9), left hand (12), right EE xyz+rot6d (9), right hand (12), base vx/vy/wz (3)`. Poses use the robot base frame. Conversion decodes HDF5 images only with `XPolicyLab.utils.process_data.decode_image_bit`, preserving RGB for both legacy and marked JPEG buffers. Training uses the resulting 224×224 RGB videos. State and action normalization statistics are computed separately.

## Training

The standard wrapper selects the MobileArena EE reader and keeps the original 80-dimensional `dual_system` model. Set the converted dataset and the OpenWAM Alpha foundation checkpoint explicitly:

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> ee <seed> <gpu_ids> [num_gpus] [Hydra overrides...]

OPENWAM_DATASET_DIR="$PWD/data/mobile_arena_ee" \
OPENWAM_FOUNDATION_CKPT=/path/to/OpenWAM-Alpha-Pretrain-Foundation-Model \
bash train.sh MobileArena openwam mobile_arena ee 0 0,1 2
```

`OPENWAM_BATCH_SIZE` defaults to 16 and `OPENWAM_MAX_STEPS` to 60000, matching the source MobileArena training entry. `OPENWAM_OUTPUT_PATH` overrides the output root; otherwise training writes beneath `checkpoints/<bench>-<ckpt>-<env>-ee-<seed>/`. OpenWAM may create a timestamped run directory inside that root; pass the directory containing `config.yaml`, `normalization_stats.npy` and checkpoint files to evaluation. Additional Hydra overrides are forwarded unchanged, including `training.resume_ckpt_path=...` for resuming a run.

`train_mobile_arena_openwam_ee.sh` also remains available as the direct source-style entry. It accepts Hydra overrides and reads `OPENWAM_GPUS`, `OPENWAM_PYTHON`, `OPENWAM_DATASET_DIR` (or `MOBILE_ARENA_DATASET_DIR`), `OPENWAM_FOUNDATION_CKPT`, and `OPENWAM_OUTPUT_PATH`.

## Evaluation

Use `env_cfg_type=mobile_arena`, `action_type=ee`, and an **EE45-trained MobileArena checkpoint**. Images are already RGB arrays when they reach `model.py`. The adapter mirrors training resize behavior (AREA to 224×224, then LANCZOS to each camera tile), keeps the raw instruction, and returns base-frame EE poses, both dexterous hands, and base velocities. Unmodeled `vz`, `wx` and `wy` are zero commands.

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> mobile_arena ee <seed> <policy_gpu> <sim_gpu> <policy_env> <sim_env>

OPENWAM_CKPT_DIR=/path/to/MobileArena-openwam-ee/run \
bash eval.sh MobileArena place_fruit_basket openwam mobile_arena ee 0 0 1 openwam RoboDojo
```

The server accepts either a conda environment name or an absolute environment prefix. The client uses the shared `utils/setup_env_client.sh` entry and the simulation scripts in the parent workspace. For unified RoboDojo orchestration, pass this directory as `--policy-dir` to the existing simulation CLI.

## Configuration

- `ckpt_dir` / `OPENWAM_CKPT_DIR`: trained directory. Otherwise the shared checkpoint resolver accepts a path as `ckpt_name`, the standard run name, or `checkpoints/<ckpt_name>`.
- `checkpoint_file` / `OPENWAM_CHECKPOINT_FILE`: optional checkpoint filename; unset loads the highest step.
- `openwam_root` / `OPENWAM_ROOT`: source tree; the server defaults to the vendored `OpenWAM/` tree.
- `openwam_deploy_config` / `OPENWAM_DEPLOY_CONFIG`: optional deployment YAML.
- `replan_steps`: executed steps per action chunk; null follows the checkpoint deployment horizon.
- `eval_batch: true`: one `generate_batch` call for the active environments.
- `allow_dummy_policy` / `OPENWAM_ALLOW_DUMMY_POLICY`: transport testing only; false for real evaluation.

Compilation, DiT cache and video decoding are disabled, and the synchronous executor is selected, matching the source MobileArena batched inference behavior. The checkpoint must declare `dataloader.type=mobile_arena_ee`, `dataloader.action_mode=mobile_arena_ee`, model action dimension 80, and separate `mobile_arena_ee_state` statistics.
