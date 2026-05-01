#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./open_case_in_slicer.sh IOG11
#   ./open_case_in_slicer.sh IOG11 --modality T1T2
#   ./open_case_in_slicer.sh IOG11 --seg-root path/to/crossval_results...
#
# Slicer executable (provided by you):
SLICER_BIN="/home/fffeelipe/Apps/Slicer-5.10.0-linux-amd64/Slicer"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <CASE_ID> [--modality T1|T2|T1T2] [--no-seg] [--seg-root <path>]" >&2
  exit 2
fi

CASE_ID="$1"
shift

exec "$SLICER_BIN" \
  --no-splash \
  --python-script "$SCRIPT_DIR/slicer_open_case.py" \
  -- "$CASE_ID" --workspace "$SCRIPT_DIR" "$@"

