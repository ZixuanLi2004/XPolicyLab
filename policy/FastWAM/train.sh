#!/bin/bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}
if [[ $# -ge 7 ]]; then
    num_gpus=${7}
elif [[ "${gpu_id}" == *,* ]]; then
    IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
    num_gpus=${#gpu_ids[@]}
else
    num_gpus=1
fi
train_seed=${seed}
if [[ "${train_seed}" -le 0 ]]; then
    train_seed=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT_DIR="${PROJECT_DIR}"
POLICY_DIR="${SCRIPT_DIR}"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"
UTILS_DIR="${ROOT_DIR}/utils"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_DIR}/checkpoints"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT_DIR}/..:${FASTWAM_DIR}:${FASTWAM_DIR}/src:${PYTHONPATH:-}"

if [[ "${FASTWAM_MOBILE_ARENA:-${FASTWAM_MIVERSE:-true}}" == "true" ]]; then
    action_dim=45
    task_name="mobile_arena_uncond_3cam_384_1e-4"
else
    action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
    task_name="robotwin_uncond_3cam_384_1e-4"
fi
# Default dataset_id is the 4-tuple data_key. Set FASTWAM_DATASET_ID to point
# at a differently named dataset without changing the train.sh argument shape.
data_key="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
if [[ "${FASTWAM_MOBILE_ARENA:-${FASTWAM_MIVERSE:-true}}" == "true" ]]; then
    dataset_id="${FASTWAM_DATASET_ID:-miverse_data}"
else
    dataset_id="${FASTWAM_DATASET_ID:-${data_key}}"
fi
if [[ "${FASTWAM_MOBILE_ARENA:-${FASTWAM_MIVERSE:-true}}" == "true" ]]; then
    # MobileArena is a shared LeRobot v3.0 dataset used by Pi0.5, GalaxeaVLA,
    # and InternVLA. FastWAM keeps its own processor but reads this same repo.
    dataset_dir="${FASTWAM_DATASET_DIR:-${POLICY_DIR}/../Pi_05/data/${dataset_id}}"
    stats_path="${FASTWAM_STATS_PATH:-${POLICY_DIR}/data/${dataset_id}/dataset_stats.json}"
else
    converted_root="${POLICY_DIR}/data/${dataset_id}"
    dataset_dir="${converted_root}/lerobot"
    stats_path="${converted_root}/dataset_stats.json"
fi
text_cache_dir="${FASTWAM_DIR}/data/text_embeds_cache/xpolicylab/${dataset_id}"
ckpt_setting="${FASTWAM_CKPT_SETTING:-${data_key}-${seed}}"
action_dit="${FASTWAM_DIR}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
batch_size="${FASTWAM_BATCH_SIZE:-16}"
gradient_accumulation_steps="${FASTWAM_GRADIENT_ACCUMULATION_STEPS:-1}"
num_workers="${FASTWAM_NUM_WORKERS:-8}"
# The official task config uses five epochs. Environment overrides remain available
# for an explicit max-step run without changing the checked-in default.
num_epochs_override="${FASTWAM_NUM_EPOCHS:-5}"
max_steps_override="${FASTWAM_MAX_STEPS:-}"
mot_checkpoint_override="${FASTWAM_MOT_CHECKPOINT_MIXED_ATTN:-}"

if [[ ! -d "${dataset_dir}/meta" ]]; then
    echo "[ERROR] LeRobot dataset not found: ${dataset_dir}/meta"
    echo "Prepare a LeRobot v3.0 dataset under ${dataset_dir} and dataset_stats.json."
    if [[ -n "${FASTWAM_DATASET_ID:-}" ]]; then
        echo "FASTWAM_DATASET_ID=${FASTWAM_DATASET_ID}"
    fi
    exit 1
fi

# With a fresh converted dataset, use the upstream statistics computation and
# save the resulting JSON beside this run's checkpoints for evaluation.
if [[ ! -f "${stats_path}" ]]; then
    stats_path=null
fi

if [[ ! -f "${action_dit}" ]]; then
    echo "[ERROR] Missing ActionDiT backbone: ${action_dit}"
    echo "Run in the FastWAM policy environment:"
    echo "  cd ${FASTWAM_DIR}"
    echo "  python scripts/preprocess_action_dit_backbone.py --model-config configs/model/fastwam.yaml --output ${action_dit} --device cuda --dtype bfloat16"
    exit 1
fi

if [[ ! -d "${text_cache_dir}" || -z "$(find "${text_cache_dir}" -name '*.pt' -print -quit 2>/dev/null)" ]]; then
    echo "[ERROR] Missing T5 text embedding cache: ${text_cache_dir}"
    echo "Precompute it with the upstream script in the FastWAM policy environment:"
    echo "  cd ${FASTWAM_DIR}"
    echo "  python scripts/precompute_text_embeds.py \\"
    echo "    task=${task_name} \\"
    echo "    data.train.dataset_dirs=[${dataset_dir}] \\"
    echo "    data.val.dataset_dirs=[${dataset_dir}] \\"
    echo "    data.train.text_embedding_cache_dir=${text_cache_dir} \\"
    echo "    data.val.text_embedding_cache_dir=${text_cache_dir}"
    exit 1
fi

cd "${FASTWAM_DIR}"

train_common=(
    "task=${task_name}"
    "seed=${train_seed}"
    "batch_size=${batch_size}"
    "gradient_accumulation_steps=${gradient_accumulation_steps}"
    "num_workers=${num_workers}"
    "num_epochs=${num_epochs_override}"
    "data.train.dataset_dirs=[${dataset_dir}]"
    "data.val.dataset_dirs=[${dataset_dir}]"
    "data.train.text_embedding_cache_dir=${text_cache_dir}"
    "data.val.text_embedding_cache_dir=${text_cache_dir}"
    "data.train.pretrained_norm_stats=${stats_path}"
    "data.val.pretrained_norm_stats=${stats_path}"
    "data.train.shape_meta.action.0.raw_shape=${action_dim}"
    "data.train.shape_meta.action.0.shape=${action_dim}"
    "data.train.shape_meta.state.0.raw_shape=${action_dim}"
    "data.train.shape_meta.state.0.shape=${action_dim}"
    "data.val.shape_meta.action.0.raw_shape=${action_dim}"
    "data.val.shape_meta.action.0.shape=${action_dim}"
    "data.val.shape_meta.state.0.raw_shape=${action_dim}"
    "data.val.shape_meta.state.0.shape=${action_dim}"
    "data.train.processor.action_output_dim=${action_dim}"
    "data.train.processor.proprio_output_dim=${action_dim}"
    "data.val.processor.action_output_dim=${action_dim}"
    "data.val.processor.proprio_output_dim=${action_dim}"
    "output_dir=${POLICY_DIR}/checkpoints/${ckpt_setting}"
)
if [[ -n "${max_steps_override}" ]]; then
    train_common+=("max_steps=${max_steps_override}")
fi
if [[ -n "${mot_checkpoint_override}" ]]; then
    train_common+=("model.mot_checkpoint_mixed_attn=${mot_checkpoint_override}")
fi

bash scripts/train_zero1.sh "${num_gpus}" "${train_common[@]}"
