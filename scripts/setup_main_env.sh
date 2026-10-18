#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Override these when the environment names or Conda installation differ.
MAIN_ENV_NAME="${MAIN_ENV_NAME:-world-hand-gen}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
THIRD_PARTY_ROOT="${WHG_THIRD_PARTY_ROOT:-$ROOT/third_party}"

# shellcheck source=scripts/_conda_helpers.sh
source "$SCRIPT_DIR/_conda_helpers.sh"
require_conda
"$SCRIPT_DIR/bootstrap_third_party.sh"

if [[ ! -d "$THIRD_PARTY_ROOT/vggt-omega" ]]; then
    printf 'error: VGGT-Omega source checkout is unavailable after bootstrap\n' >&2
    exit 1
fi

ensure_conda_env "$MAIN_ENV_NAME" 3.10

run_in_conda_env "$MAIN_ENV_NAME" python -m pip install --upgrade pip setuptools wheel

TORCH_PACKAGES=("torch>=2.3" "torchvision>=0.18")
if [[ -n "$TORCH_INDEX_URL" ]]; then
    run_in_conda_env "$MAIN_ENV_NAME" python -m pip install \
        "${TORCH_PACKAGES[@]}" --index-url "$TORCH_INDEX_URL"
else
    run_in_conda_env "$MAIN_ENV_NAME" python -m pip install "${TORCH_PACKAGES[@]}"
fi

# These packages are imported by the project or its data/model wrappers but are
# intentionally kept outside pyproject.toml because the upstream environments
# have incompatible detector and vision dependencies.
run_in_conda_env "$MAIN_ENV_NAME" python -m pip install \
    'numpy<2' Pillow scipy opencv-python projectaria-tools smplx yacs pytest matplotlib

# Training imports VGGT in the main process, so it belongs in this environment
# as well as in the separate evaluation worker environment.
run_in_conda_env "$MAIN_ENV_NAME" python -m pip install -e "$THIRD_PARTY_ROOT/vggt-omega"
run_in_conda_env "$MAIN_ENV_NAME" python -m pip install -e "$ROOT"

MAIN_PYTHON="$(conda_python_path "$MAIN_ENV_NAME")"
printf '\nConfigured project environment: %s\n' "$MAIN_ENV_NAME"
printf 'Python: %s\n' "$MAIN_PYTHON"
printf '\nSet these variables for runtime workers if they use separate environments:\n'
printf 'export WHG_HAMER_PYTHON=%q\n' "$MAIN_PYTHON"
printf 'export WHG_DETECTOR_PYTHON=%q\n' "$MAIN_PYTHON"
printf 'export WHG_VGGT_PYTHON=%q\n' "$MAIN_PYTHON"
printf 'Then set WHG_DATA_ROOT, WHG_CACHE_ROOT, and WHG_EXPERIMENT_ROOT for this machine.\n'
