#!/usr/bin/env bash
set -euo pipefail

# Usage: bash process_data.sh <MiVerse HDF5 root> [converted dataset directory] [converter flags...]
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir=${1:?Usage: bash process_data.sh <MiVerse HDF5 root> [converted dataset directory] [converter flags...]}
dataset_dir=${2:-${OPENWAM_DATASET_DIR:-${POLICY_DIR}/data/miverse_ee}}
shift
if [[ $# -gt 0 ]]; then shift; fi
exec "${OPENWAM_PYTHON:-python}" "${POLICY_DIR}/convert_miverse_openwam_ee.py" \
    --src "${source_dir}" --out "${dataset_dir}" "$@"
