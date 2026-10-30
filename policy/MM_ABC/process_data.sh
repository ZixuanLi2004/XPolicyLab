#!/bin/bash
set -euo pipefail

bench_name=${1:?usage: process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]}
ckpt_name=${2:-}
env_cfg_type=${3:-m92uw}
action_type=${4:-joint}
expert_data_num=${5:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMABC_ROOT=${MMABC_ROOT:-"${POLICY_DIR}/MM_ABC"}
# shellcheck source=resolve_python.sh
source "${POLICY_DIR}/resolve_python.sh"
PY=${MMABC_PYTHON:-$(resolve_python "${MMABC_ENV:-none}")}

if [[ ( "${bench_name}" != "MobileArena" && "${bench_name}" != "MiVerse" ) || ( "${env_cfg_type}" != "m92uw" && "${env_cfg_type}" != "mobile_arena" && "${env_cfg_type}" != "miverse" ) || "${action_type}" != "joint" ]]; then
    echo "[MM_ABC] supports bench_name=MobileArena env_cfg_type=m92uw|mobile_arena action_type=joint" >&2
    exit 1
fi

args=(--workers "${WORKERS:-64}")
raw_source="${MOBILE_ARENA_SRC:-${MIVERSE_SRC:-}}"
[[ -n "${raw_source}" ]] && args+=(--src "${raw_source}")
[[ -n "${EPISODE_SUBDIR:-}" ]] && args+=(--episode-subdir "${EPISODE_SUBDIR}")
[[ -n "${expert_data_num}" ]] && args+=(--max-episodes "${expert_data_num}")

echo "[MM_ABC] converting ${bench_name} (${env_cfg_type}/${action_type}) with ${PY}"
cd "${MMABC_ROOT}"
OMP_NUM_THREADS=1 "${PY}" scripts/convert_mobile_arena.py "${args[@]}"
OMP_NUM_THREADS=1 exec "${PY}" scripts/build_mobile_arena_c80.py
