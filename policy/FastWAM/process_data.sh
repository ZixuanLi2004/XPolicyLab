#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 MiVerse <ckpt_name> miverse joint [max_episodes] [raw_source_root]" >&2
    exit 1
fi
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"
dataset_id="${FASTWAM_DATASET_ID:-miverse_data}"
dataset_dir="${FASTWAM_DATASET_DIR:-${POLICY_DIR}/../Pi_05/data/${dataset_id}}"
text_cache_dir="${FASTWAM_DIR}/data/text_embeds_cache/xpolicylab/${dataset_id}"

# Pi_05 and FastWAM consume the same RGB 224x224, joint45 LeRobot v3 dataset.
# Conversion runs in the Pi_05 environment; T5 preparation uses the active
# FastWAM environment. An existing dataset can be reused without conversion.
if [[ ! -f "${dataset_dir}/meta/info.json" ]]; then
    MIVERSE_REPO_ID="$(basename "${dataset_dir}")" \
        OPENPI_HF_LEROBOT_HOME="$(dirname "${dataset_dir}")" \
        bash "${POLICY_DIR}/../Pi_05/process_data.sh" "$@"
fi
export PYTHONPATH="${FASTWAM_DIR}:${FASTWAM_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_DIR}/checkpoints"
cd "${FASTWAM_DIR}"
exec python scripts/precompute_text_embeds.py \
    task=miverse_uncond_3cam_384_1e-4 \
    "data.train.dataset_dirs=[${dataset_dir}]" \
    "data.val.dataset_dirs=[${dataset_dir}]" \
    "data.train.text_embedding_cache_dir=${text_cache_dir}" \
    "data.val.text_embedding_cache_dir=${text_cache_dir}"
