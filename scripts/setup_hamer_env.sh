#!/usr/bin/env bash
set -euo pipefail

# Read the upstream installation prerequisite before continuing:
# https://github.com/geopavlakos/hamer/issues/12
# The issue documents a required dependency/workaround that depends on the
# target CUDA and PyTorch installation. Resolve it before running this script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

HAMER_ENV_NAME="${HAMER_ENV_NAME:-hamer}"
# HaMeR's README uses the CUDA 11.7 wheel index. Set this to an empty string
# to use the default PyPI index, or replace it with the index for your driver.
HAMER_TORCH_INDEX_URL="${HAMER_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu117}"
THIRD_PARTY_ROOT="${WHG_THIRD_PARTY_ROOT:-$ROOT/third_party}"
HAMER_DIR="$THIRD_PARTY_ROOT/hamer"

# shellcheck source=scripts/_conda_helpers.sh
source "$SCRIPT_DIR/_conda_helpers.sh"
require_conda
"$SCRIPT_DIR/bootstrap_third_party.sh"

if [[ ! -d "$HAMER_DIR" ]]; then
    printf 'error: HaMeR source checkout is unavailable after bootstrap\n' >&2
    exit 1
fi
if [[ ! -d "$HAMER_DIR/third-party/ViTPose" ]]; then
    printf 'error: ViTPose source checkout is unavailable after bootstrap\n' >&2
    exit 1
fi

ensure_conda_env "$HAMER_ENV_NAME" 3.10
run_in_conda_env "$HAMER_ENV_NAME" python -m pip install --upgrade pip setuptools wheel

if [[ -n "$HAMER_TORCH_INDEX_URL" ]]; then
    run_in_conda_env "$HAMER_ENV_NAME" python -m pip install \
        torch torchvision --index-url "$HAMER_TORCH_INDEX_URL"
else
    run_in_conda_env "$HAMER_ENV_NAME" python -m pip install torch torchvision
fi

run_in_conda_env "$HAMER_ENV_NAME" python -m pip install -e "${HAMER_DIR}[all]"
run_in_conda_env "$HAMER_ENV_NAME" python -m pip install -v -e "$HAMER_DIR/third-party/ViTPose"

HAMER_PYTHON="$(conda_python_path "$HAMER_ENV_NAME")"
printf '\nConfigured HaMeR environment: %s\n' "$HAMER_ENV_NAME"
printf 'Python: %s\n' "$HAMER_PYTHON"
printf 'Model assets are not downloaded by this script.\n'
printf 'After reviewing upstream licenses, place or fetch the required assets at the paths reported by scripts/check_environment.sh.\n'
printf '\nSet this worker path before training/evaluation:\n'
printf 'export WHG_HAMER_PYTHON=%q\n' "$HAMER_PYTHON"
printf 'export WHG_DETECTOR_PYTHON=%q\n' "$HAMER_PYTHON"
