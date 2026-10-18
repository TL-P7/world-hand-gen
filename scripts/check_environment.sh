#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MAIN_ENV_NAME="${MAIN_ENV_NAME:-world-hand-gen}"
HAMER_ENV_NAME="${HAMER_ENV_NAME:-hamer}"
VGGT_ENV_NAME="${VGGT_ENV_NAME:-vggt-omega}"

mapfile -t CONFIG_PATHS < <(
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -c '
from src.config import (
    CACHE_ROOT, DATA_ROOT, DETECTOR_CHECKPOINT, DETECTOR_PYTHON, EGODEX_ROOT,
    EXPERIMENT_ROOT, HAMER_CHECKPOINT, HAMER_PYTHON, HAMER_REPOSITORY,
    MANO_ROOT, VGGT_CHECKPOINT, VGGT_PYTHON, VGGT_REPOSITORY,
)
for name, path in (("WHG_DATA_ROOT", DATA_ROOT), ("WHG_EGODEX_ROOT", EGODEX_ROOT),
                   ("WHG_CACHE_ROOT", CACHE_ROOT), ("WHG_EXPERIMENT_ROOT", EXPERIMENT_ROOT),
                   ("WHG_MANO_ROOT", MANO_ROOT), ("WHG_HAMER_REPOSITORY", HAMER_REPOSITORY),
                   ("WHG_HAMER_CHECKPOINT", HAMER_CHECKPOINT),
                   ("WHG_VGGT_REPOSITORY", VGGT_REPOSITORY),
                   ("WHG_VGGT_CHECKPOINT", VGGT_CHECKPOINT),
                   ("WHG_DETECTOR_CHECKPOINT", DETECTOR_CHECKPOINT),
                   ("WHG_HAMER_PYTHON", HAMER_PYTHON),
                   ("WHG_DETECTOR_PYTHON", DETECTOR_PYTHON),
                   ("WHG_VGGT_PYTHON", VGGT_PYTHON)):
    print(f"{name}\t{path}")
'
)

declare -A CONFIG
for line in "${CONFIG_PATHS[@]}"; do
    key="${line%%$'\t'*}"
    CONFIG["$key"]="${line#*$'\t'}"
done

# shellcheck source=scripts/_conda_helpers.sh
source "$SCRIPT_DIR/_conda_helpers.sh"

status=0

check_path() {
    local path="$1" kind="$2" hint="$3"
    if [[ "$kind" == file && -f "$path" ]] || [[ "$kind" == dir && -d "$path" ]]; then
        printf 'OK      %s\n' "$path"
    else
        printf 'MISSING %s\n        set %s or provide this asset\n' "$path" "$hint" >&2
        status=1
    fi
}

check_git_checkout() {
    local path="$1" label="$2"
    if [[ -d "$path/.git" || -f "$path/.git" ]]; then
        printf 'OK      %s checkout: %s\n' "$label" "$path"
    else
        printf 'MISSING %s Git checkout: %s\n        run scripts/bootstrap_third_party.sh\n' "$label" "$path" >&2
        status=1
    fi
}

check_conda_env() {
    local name="$1"
    if run_in_conda_env "$name" python --version >/dev/null 2>&1; then
        printf 'OK      Conda environment %s (%s)\n' "$name" "$(conda_python_path "$name")"
    else
        printf 'MISSING Conda environment %s\n        run the corresponding setup script\n' "$name" >&2
        status=1
    fi
}

printf 'Resolved configuration\n'
for key in "${!CONFIG[@]}"; do
    printf '  %-28s %s\n' "$key" "${CONFIG[$key]}"
done

printf '\nConda environments\n'
if require_conda; then
    check_conda_env "$MAIN_ENV_NAME"
    check_conda_env "$HAMER_ENV_NAME"
    check_conda_env "$VGGT_ENV_NAME"
else
    status=1
fi

printf '\nThird-party source checkouts\n'
check_git_checkout "${CONFIG[WHG_HAMER_REPOSITORY]}" "HaMeR"
check_git_checkout "${CONFIG[WHG_VGGT_REPOSITORY]}" "VGGT-Omega"
if [[ -d "${CONFIG[WHG_MANO_ROOT]}" ]]; then
    printf 'OK      MANO asset directory: %s\n' "${CONFIG[WHG_MANO_ROOT]}"
else
    printf 'MISSING MANO asset directory: %s\n        set WHG_MANO_ROOT or run scripts/bootstrap_third_party.sh\n' "${CONFIG[WHG_MANO_ROOT]}" >&2
    status=1
fi
check_git_checkout "${CONFIG[WHG_HAMER_REPOSITORY]}/third-party/ViTPose" "ViTPose"

printf '\nRuntime assets\n'
check_path "${CONFIG[WHG_MANO_ROOT]}/mano/MANO_LEFT.pkl" file WHG_MANO_ROOT
check_path "${CONFIG[WHG_MANO_ROOT]}/mano/MANO_RIGHT.pkl" file WHG_MANO_ROOT
check_path "${CONFIG[WHG_DATA_ROOT]}/clip_splits.json" file WHG_DATA_ROOT
check_path "${CONFIG[WHG_DATA_ROOT]}/train_aria" dir WHG_DATA_ROOT
check_path "${CONFIG[WHG_HAMER_REPOSITORY]}/_DATA/data/mano/MANO_RIGHT.pkl" file WHG_HAMER_REPOSITORY
check_path "${CONFIG[WHG_HAMER_REPOSITORY]}/_DATA/data/mano_mean_params.npz" file WHG_HAMER_REPOSITORY
check_path "${CONFIG[WHG_HAMER_CHECKPOINT]}" file WHG_HAMER_CHECKPOINT
check_path "${CONFIG[WHG_HAMER_REPOSITORY]}/_DATA/hamer_ckpts/model_config.yaml" file WHG_HAMER_REPOSITORY
check_path "${CONFIG[WHG_HAMER_REPOSITORY]}/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth" file WHG_HAMER_REPOSITORY
check_path "${CONFIG[WHG_DETECTOR_CHECKPOINT]}" file WHG_DETECTOR_CHECKPOINT
check_path "${CONFIG[WHG_VGGT_CHECKPOINT]}" file WHG_VGGT_CHECKPOINT

printf '\nWorker interpreters\n'
for key in WHG_HAMER_PYTHON WHG_DETECTOR_PYTHON WHG_VGGT_PYTHON; do
    if [[ -x "${CONFIG[$key]}" ]]; then
        printf 'OK      %s: %s\n' "$key" "${CONFIG[$key]}"
    else
        printf 'MISSING %s executable: %s\n        set %s to a Python interpreter from the matching environment\n' "$key" "${CONFIG[$key]}" "$key" >&2
        status=1
    fi
done

if [[ "$status" -eq 0 ]]; then
    printf '\nEnvironment check passed.\n'
else
    printf '\nEnvironment check found missing prerequisites.\n' >&2
fi
exit "$status"
