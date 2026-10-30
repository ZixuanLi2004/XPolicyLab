#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENWAM_ROOT="${OPENWAM_ROOT:-${POLICY_DIR}/OpenWAM}"
PYTHON_BIN="$(command -v "${OPENWAM_PYTHON:-python}")"
DATASET_DIR="${OPENWAM_DATASET_DIR:-${MOBILE_ARENA_DATASET_DIR:-${POLICY_DIR}/data/mobile_arena_ee}}"
FOUNDATION_CKPT="${OPENWAM_FOUNDATION_CKPT:-${POLICY_DIR}/checkpoints/OpenWAM-Alpha-Pretrain-Foundation-Model}"
OUTPUT_PATH="${OPENWAM_OUTPUT_PATH:-${POLICY_DIR}/checkpoints/MobileArena-openwam-ee}"
GPUS="${OPENWAM_GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${OPENWAM_BATCH_SIZE:-16}"
MAX_STEPS="${OPENWAM_MAX_STEPS:-60000}"

[[ -x "${PYTHON_BIN}" ]] || { echo "missing OpenWAM python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -f "${DATASET_DIR}/meta/info.json" ]] || { echo "missing MobileArena EE dataset: ${DATASET_DIR}" >&2; exit 1; }
[[ -f "${FOUNDATION_CKPT}/config.yaml" ]] || { echo "missing OpenWAM foundation checkpoint: ${FOUNDATION_CKPT}" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${OPENWAM_ROOT}:${PYTHONPATH:-}"
IFS=',' read -r -a selected_gpus <<< "${GPUS}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#selected_gpus[@]}}"

cd "${OPENWAM_ROOT}"
exec bash scripts/train.sh \
  model=dual_system \
  dataloader=mobile_arena_ee \
  dataloader.dataset_dir="${DATASET_DIR}" \
  training.finetune_ckpt_path="${FOUNDATION_CKPT}" \
  training.output_path="${OUTPUT_PATH}" \
  training.batch_size="${BATCH_SIZE}" \
  training.num_epochs=null \
  training.max_steps="${MAX_STEPS}" \
  training.gradient_accumulation_steps=1 \
  "$@"
