#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

VGGT_ENV_NAME="${VGGT_ENV_NAME:-vggt-omega}"
THIRD_PARTY_ROOT="${WHG_THIRD_PARTY_ROOT:-$ROOT/third_party}"
VGGT_DIR="$THIRD_PARTY_ROOT/vggt-omega"

# shellcheck source=scripts/_conda_helpers.sh
source "$SCRIPT_DIR/_conda_helpers.sh"
require_conda
"$SCRIPT_DIR/bootstrap_third_party.sh"

if [[ ! -d "$VGGT_DIR" ]]; then
    printf 'error: VGGT-Omega source checkout is unavailable after bootstrap\n' >&2
    exit 1
fi
if [[ ! -f "$VGGT_DIR/requirements.txt" ]]; then
    printf 'error: missing VGGT-Omega requirements.txt\n' >&2
    exit 1
fi

ensure_conda_env "$VGGT_ENV_NAME" 3.10
run_in_conda_env "$VGGT_ENV_NAME" python -m pip install --upgrade pip setuptools wheel
run_in_conda_env "$VGGT_ENV_NAME" python -m pip install -r "$VGGT_DIR/requirements.txt"
run_in_conda_env "$VGGT_ENV_NAME" python -m pip install -e "$VGGT_DIR"

VGGT_PYTHON="$(conda_python_path "$VGGT_ENV_NAME")"
printf '\nConfigured VGGT-Omega environment: %s\n' "$VGGT_ENV_NAME"
printf 'Python: %s\n' "$VGGT_PYTHON"
printf 'The checkpoint is not downloaded by this script. Set WHG_VGGT_CHECKPOINT to its licensed local path.\n'
printf '\nSet this worker path before training/evaluation:\n'
printf 'export WHG_VGGT_PYTHON=%q\n' "$VGGT_PYTHON"
