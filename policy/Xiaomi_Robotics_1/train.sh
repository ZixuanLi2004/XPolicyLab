#!/usr/bin/env bash
# Usage: bash train.sh MobileArena <ckpt_name> mobile_arena ee <seed> <gpu_ids> [hydra_overrides...]
# Run in the installed mibot environment. Multi-node: set WORLD_SIZE, RANK,
# MASTER_ADDR and MASTER_PORT on every node; WORLD_SIZE is the node count here.
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 MobileArena <ckpt_name> mobile_arena ee <seed> <gpu_ids> [hydra_overrides...]" >&2
    exit 1
fi
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6
if [[ ( "${bench_name}" != MobileArena && "${bench_name}" != MiVerse ) || "${action_type}" != ee ]]; then
    echo "Xiaomi_Robotics_1 supports MobileArena end-effector training (MobileArena ... ee)." >&2
    exit 1
fi
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR1_DIR="${POLICY_DIR}/xiaomi_robotics_1/xr1"
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
output_dir="${MOBILE_ARENA_OUTPUT_ROOT:-${MIVERSE_OUTPUT_ROOT:-${POLICY_DIR}/data/miverse_native}}"
config_name="${DATA_CONFIG_NAME:-mobile_arena_native}"
config_path="${XR1_DIR}/configs/data/${config_name}.yaml"
pretrained_path="${PRETRAINED_PATH:-${POLICY_DIR}/checkpoints/pretrained_ckpt/model_states.pt}"
run_root="${RUN_ROOT:-${POLICY_DIR}/train_runs/${ckpt_setting}}"
project="${PROJECT:-xr1}"
artifact_dir="${run_root}/project_${project}/${ckpt_setting}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
for required in "${config_path}" "${pretrained_path}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing file: ${required}" >&2
        exit 1
    fi
done
if [[ ! -d "${output_dir}/data" ]]; then
    echo "Converted data missing: ${output_dir}/data; run process_data.sh first." >&2
    exit 1
fi
if [[ -e "${ckpt_dir}" && ! -L "${ckpt_dir}" ]]; then
    echo "Refusing to replace checkpoint directory: ${ckpt_dir}" >&2
    exit 1
fi
mkdir -p "${artifact_dir}" "${POLICY_DIR}/checkpoints"
ln -sfn "${artifact_dir}" "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export RESOURCE_GPU="${RESOURCE_GPU:-$(tr ',' '\n' <<< "${gpu_id}" | sed '/^$/d' | wc -l | xargs)}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MAX_LENGTH="${MAX_LENGTH:-20000}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${run_root}}"
cd "${XR1_DIR}"
exec bash scripts/train.sh \
    "data=${config_name}" "model=posttrain_mobile_arena" \
    "data.params.train_datasets.paths=[${output_dir}/data]" \
    "data.params.train_datasets.batch_size=${BATCH_SIZE:-8}" \
    "data.params.num_workers=${NUM_WORKERS:-16}" \
    "trainer.project=${project}" \
    "trainer.exp_name=${ckpt_setting}" \
    "trainer.default_root_dir=${run_root}" \
    "trainer.seed=${seed}" "trainer.num_nodes=${WORLD_SIZE}" \
    "trainer.max_steps=${MAX_STEPS:-60000}" \
    "trainer.save_interval=${SAVE_INTERVAL:-10000}" \
    "trainer.accumulate_grad_batches=1" \
    "trainer.scheduler.params.max_lr=${MAX_LR:-1e-5}" \
    "model.params.pretrained=${pretrained_path}" \
    "model.params.model.async_train=true" \
    "hydra.run.dir=${run_root}/hydra/rank_\${oc.env:RANK,0}" \
    "$@"
