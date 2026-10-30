# Xiaomi_Robotics_1 environment

Run `MIBOT_CONDA_ENV=mibot bash install.sh` from this directory. The script creates
a Python 3.12 policy environment, installs the source PyTorch 2.8/cu128 recipe,
FlashAttention, XR1 training dependencies, and HDF5/video conversion dependencies.
It does not install or alter the simulator environment.

The source model initializes `Qwen/Qwen3-VL-4B-Instruct` configuration and tokenizer
assets through Hugging Face. Prepare that cache on offline hosts (`HF_HOME` can
select its location). `vlm_processor_path` in `deploy.yml` overrides the inference
processor location. Model weights are supplied separately:

- Initial training: set `PRETRAINED_PATH` to the converted Xiaomi pretrained
  `model_states.pt`; `xiaomi_robotics_1/xr1/tools/weight_convert.py` provides the
  upstream conversion entry.
- MobileArena evaluation: set `XR1_CKPT_DIR` to the current official-state training
  artifact directory containing `config.py` and `last.ckpt/checkpoint/`.

Activate `mibot` before running `process_data.sh` or `train.sh`. Evaluation accepts
a conda environment name or an environment directory as its policy-environment
argument. See [README.md](README.md) for the conversion, training, and eval commands.
