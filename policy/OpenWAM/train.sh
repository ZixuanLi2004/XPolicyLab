#!/usr/bin/env bash
set -euo pipefail

# Standard XPolicyLab arguments; extra arguments are Hydra overrides.
bench_name=${1:?Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [num_gpus] [Hydra overrides...]}
ckpt_name=${2:?}
env_cfg_type=${3:?}
action_type=${4:?}
seed=${5:?}
gpu_ids=${6:?}
shift 6
if [[ "${action_type}" != "ee" ]]; then
    echo '[OpenWAM] MiVerse OpenWAM requires action_type=ee.' >&2
    exit 1
fi
if [[ $# -gt 0 && "$1" =~ ^[1-9][0-9]*$ ]]; then
    export NPROC_PER_NODE=$1
    shift
else
    IFS=',' read -r -a selected_gpus <<< "${gpu_ids}"
    export NPROC_PER_NODE=${#selected_gpus[@]}
fi
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENWAM_GPUS="${gpu_ids}"
export OPENWAM_OUTPUT_PATH="${OPENWAM_OUTPUT_PATH:-${POLICY_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}}"
exec bash "${POLICY_DIR}/train_miverse_openwam_ee.sh" "project.seed=${seed}" "$@"
