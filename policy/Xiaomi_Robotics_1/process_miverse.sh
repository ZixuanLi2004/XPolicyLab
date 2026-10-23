#!/usr/bin/env bash
# Keep the upstream MiVerse conversion entry as an alias.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/process_data.sh" "$@"
