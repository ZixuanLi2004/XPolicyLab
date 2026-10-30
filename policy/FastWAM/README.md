# FastWAM for MobileArena

**Contributor:** RoboDojo Team | **Original code:** https://github.com/yuantianyuan01/FastWAM

This adapter includes MobileArena evaluation, data preparation and training. `FastWAM/` contains the vendored implementation. It uses the same M92UW joint45 dataset as Pi_05. The [XPolicyLab README](../../README.md) describes shared launch conventions.

## Installation

```bash
cd /personal/lzx/MobileArena/XPolicyLab/policy/FastWAM
bash install.sh
conda activate fastwam
```

Read [INSTALLATION.md](INSTALLATION.md) for pretrained Wan/T5 weights and ActionDiT preprocessing. Downloaded weights, datasets, environment directories and logs are not part of the migration. The released RoboTwin task checkpoint/statistics are examples, not MobileArena evaluation weights.

## Data Processing

```bash
# Activate FastWAM; Pi_05's interpreter must also be installed for conversion.
MOBILE_ARENA_RAW_ROOT=/path/to/raw/MobileArena \
  bash process_data.sh MobileArena mobile_arena_full mobile_arena joint
```

This reuses `../Pi_05/data/mobile_arena_data` when it exists. Otherwise it calls Pi_05's MobileArena HDF5 converter in `openpi/.venv`, then runs FastWAM's T5 text-embedding preparation in the active FastWAM environment. Optional final arguments are `max_episodes` and `raw_source_root`. `OPENPI_PYTHON` selects an existing Pi_05 interpreter.

The shared dataset is **LeRobot v3.0**, with 45D `observation.state`/`action` and RGB image keys `observation.images.cam_head/cam_left_wrist/cam_right_wrist`. The custom MobileArena converter selects 45 controllable dimensions and produces 224×224 images; FastWAM then applies its original 240×320 resize and three-camera mosaic. No additional channel swap is performed.

`FASTWAM_DATASET_ID` defaults to `mobile_arena_data`; `FASTWAM_DATASET_DIR` overrides the dataset directory. T5 embeddings go to `FastWAM/data/text_embeds_cache/xpolicylab/<dataset_id>/`. If existing normalization stats are available, use `FASTWAM_STATS_PATH`; otherwise training computes statistics with the upstream FastWAM dataset implementation and writes `dataset_stats.json` into the run directory.

## Training

```bash
bash train.sh MobileArena mobile_arena_full mobile_arena joint 0 0,1,2,3
```

The optional seventh argument is the process count. Defaults retain the source MobileArena recipe: task `mobile_arena_uncond_3cam_384_1e-4`, batch 16, five epochs, AdamW learning rate `1e-4`, weight decay `1e-2`, cosine schedule, accumulation 1, bf16 and clipping 1.0. The action scheduler uses training/inference shift 1.0; the video scheduler uses 5.0. No extra `train_shift` deployment override is required.

Overrides include `FASTWAM_BATCH_SIZE`, `FASTWAM_GRADIENT_ACCUMULATION_STEPS`, `FASTWAM_NUM_WORKERS`, `FASTWAM_NUM_EPOCHS`, `FASTWAM_MAX_STEPS`, and `FASTWAM_MOT_CHECKPOINT_MIXED_ATTN`. Training writes `checkpoints/MobileArena-<ckpt_name>-mobile_arena-joint-<seed>/`, including the normalization statistics required for evaluation.

## Evaluation

```bash
bash eval.sh MobileArena place_fruit_basket mobile_arena_full mobile_arena joint 0 0 1 fastwam RoboDojo
```

The checkpoint argument may be a short run name, full run directory, or absolute path. The launcher selects the latest `checkpoints/weights/step_*.pt` and the run's `dataset_stats.json`. `FASTWAM_CHECKPOINT_PATH` and `FASTWAM_DATASET_STATS_PATH` override these files.

`deploy.yml` defaults to joint45, 10 denoising steps, 24 executed steps per inference chunk, CFG 1.0 and inference action shift 1.0 from the task config (`sigma_shift: null`). Use a MobileArena joint45 checkpoint and its matching stats. Old 75D checkpoints are unsupported. Unmodeled root velocity components are returned as zero; batched observations and reset are supported.

`FASTWAM_ALLOW_DUMMY_POLICY=true` is an existing wiring-debug option that returns zero actions. It does not validate trained inference. Real checkpoint loading, training and simulator evaluation require the corresponding environments and weights.
