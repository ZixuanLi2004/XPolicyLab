#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve the checkout and its importable parent independently of the CWD.
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XPL_PARENT="$(dirname "${XPL_ROOT}")"
XR1_ROOT="${SCRIPT_DIR}/xiaomi_robotics_1/xr1"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}"
echo "[SERVER] xpl_root=${XPL_ROOT}"
echo "[SERVER] deploy=${yaml_file}"

if [[ -x "${policy_conda_env}/bin/python" ]]; then
    PYTHON_BIN="${policy_conda_env}/bin/python"
else
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="python"
fi

overrides=(
    "port=${policy_server_port}"
    "host=${policy_server_host}"
    "bench_name=${bench_name}"
    "task_name=${task_name}"
    "ckpt_name=${ckpt_name}"
    "env_cfg_type=${env_cfg_type}"
    "seed=${seed}"
    "policy_name=${policy_name}"
    "action_type=${action_type}"
)
if [[ -n "${XR1_CKPT_DIR:-}" ]]; then
    overrides+=("model_dir=${XR1_CKPT_DIR}")
    echo "[SERVER] model_dir=${XR1_CKPT_DIR}"
fi

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${XPL_PARENT}:${XR1_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            "${overrides[@]}"
