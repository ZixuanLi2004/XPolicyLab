#!/usr/bin/env bash
# Convert native MiVerse HDF5 to XR1 JSON/MP4 and compute normalization statistics.
# Usage: bash process_data.sh [--source-root PATH] [--output-root PATH] [--workers N]
# Run in the installed mibot environment. Additional options go to the converter.
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
XR1_DIR="${POLICY_DIR}/xiaomi_robotics_1/xr1"
exec python "${XR1_DIR}/scripts/process_miverse.py" \
    --source-root "${MIVERSE_RAW_ROOT:-${WORKSPACE_DIR}/data/MiVerse}" \
    --output-root "${MIVERSE_OUTPUT_ROOT:-${POLICY_DIR}/data/miverse_native}" \
    --config "${MIVERSE_CONFIG_PATH:-${XR1_DIR}/configs/data/miverse_native.yaml}" \
    "$@"
