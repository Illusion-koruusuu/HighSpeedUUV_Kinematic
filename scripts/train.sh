#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR="$project_dir/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$project_dir"
exec /home/lu/code/REMUS100_HoloOcean/.venv/bin/python -m training.train "$@"

