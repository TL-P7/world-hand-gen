#!/usr/bin/env bash
set -euo pipefail

require_conda() {
    if [[ -z "${CONDA_EXE:-}" ]]; then
        CONDA_EXE="$(command -v conda || true)"
    fi
    if [[ -z "${CONDA_EXE:-}" ]]; then
        printf 'error: conda was not found in PATH\n' >&2
        return 1
    fi
    export CONDA_EXE
}

conda_env_ref() {
    local name="$1"
    if [[ -n "${CONDA_ENV_ROOT:-}" ]]; then
        printf '%s/%s' "${CONDA_ENV_ROOT%/}" "$name"
    else
        printf '%s' "$name"
    fi
}

run_in_conda_env() {
    local name="$1"
    shift
    if [[ -n "${CONDA_ENV_ROOT:-}" ]]; then
        "$CONDA_EXE" run --no-capture-output --prefix "$(conda_env_ref "$name")" "$@"
    else
        "$CONDA_EXE" run --no-capture-output --name "$name" "$@"
    fi
}

ensure_conda_env() {
    local name="$1"
    local python_version="${2:-3.10}"

    if run_in_conda_env "$name" python --version >/dev/null 2>&1; then
        printf 'Reusing conda environment: %s\n' "$(conda_env_ref "$name")"
        return
    fi

    printf 'Creating conda environment: %s\n' "$(conda_env_ref "$name")"
    if [[ -n "${CONDA_ENV_ROOT:-}" ]]; then
        "$CONDA_EXE" create --yes --prefix "$(conda_env_ref "$name")" "python=$python_version"
    else
        "$CONDA_EXE" create --yes --name "$name" "python=$python_version"
    fi
}

conda_python_path() {
    local name="$1"
    run_in_conda_env "$name" python -c 'import sys; print(sys.executable)'
}
