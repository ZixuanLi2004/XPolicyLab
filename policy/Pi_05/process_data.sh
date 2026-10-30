#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 MobileArena <ckpt_name> mobile_arena joint [max_episodes] [raw_source_root]" >&2
    exit 1
fi
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
max_episodes=${5:-}
raw_source_root=${6:-${MOBILE_ARENA_RAW_ROOT:-}}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
export PYTHONPATH="${XPL_ROOT}/..:${POLICY_DIR}/openpi/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME="${OPENPI_HF_LEROBOT_HOME:-${HF_LEROBOT_HOME:-${POLICY_DIR}/data}}"
PYTHON_BIN="${OPENPI_PYTHON:-${POLICY_DIR}/openpi/.venv/bin/python}"

if [[ "${bench_name}" != "MobileArena" || "${action_type}" != "joint" ]]; then
    echo "[Pi_05] This converter supports MobileArena joint45 data." >&2
    exit 1
fi
if [[ -n "${max_episodes}" && ! "${max_episodes}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[Pi_05] max_episodes must be a positive integer." >&2
    exit 1
fi
repo_id="${MOBILE_ARENA_REPO_ID:-mobile_arena_data}"
args=(scripts/process_mobile_arena.py --repo-id "${repo_id}")
[[ -z "${max_episodes}" ]] || args+=(--max-episodes "${max_episodes}")
[[ -z "${raw_source_root}" ]] || args+=(--source-root "${raw_source_root}")
[[ "${MOBILE_ARENA_OVERWRITE:-false}" != "true" ]] || args+=(--overwrite)
cd "${POLICY_DIR}/openpi"
"${PYTHON_BIN}" "${args[@]}"
# OpenPI saves these stats into each checkpoint's assets during training.
"${PYTHON_BIN}" scripts/compute_norm_stats.py \
    --config-name pi05_mobile_arena_full --repo-id "${repo_id}"
