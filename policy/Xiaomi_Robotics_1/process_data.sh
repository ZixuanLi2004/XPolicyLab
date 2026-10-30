#!/usr/bin/env bash
# Convert native MobileArena HDF5 to XR1 JSON/MP4 and compute normalization statistics.
# Usage: bash process_data.sh [--source-root PATH] [--output-root PATH] [--workers N]
# Run in the installed mibot environment. Additional options go to the converter.
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XR1_DIR="${POLICY_DIR}/xiaomi_robotics_1/xr1"
raw_root="$(python "${POLICY_DIR}/../../utils/benchmark_compat.py")"
exec python "${XR1_DIR}/scripts/process_mobile_arena.py" \
    --source-root "${raw_root}" \
    --output-root "${MOBILE_ARENA_OUTPUT_ROOT:-${MIVERSE_OUTPUT_ROOT:-${POLICY_DIR}/data/miverse_native}}" \
    --config "${MOBILE_ARENA_CONFIG_PATH:-${MIVERSE_CONFIG_PATH:-${XR1_DIR}/configs/data/mobile_arena_native.yaml}}" \
    "$@"
