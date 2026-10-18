#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY_ROOT="${WHG_THIRD_PARTY_ROOT:-$ROOT/third_party}"

HAMER_REPO_URL="${WHG_HAMER_REPO_URL:-https://github.com/geopavlakos/hamer.git}"
HAMER_REPO_REF="${WHG_HAMER_REPO_REF:-}"
VGGTO_REPO_URL="${WHG_VGGT_REPO_URL:-https://github.com/facebookresearch/vggt-omega.git}"
VGGTO_REPO_REF="${WHG_VGGT_REPO_REF:-}"
VITPOSE_REPO_URL="${WHG_VITPOSE_REPO_URL:-https://github.com/ViTAE-Transformer/ViTPose.git}"
VITPOSE_REPO_REF="${WHG_VITPOSE_REPO_REF:-}"

clone_or_check() {
    local destination="$1"
    local url="$2"
    local revision="$3"
    local label="$4"

    if [[ -e "$destination" && ! -d "$destination" ]]; then
        printf 'error: %s path exists but is not a directory: %s\n' "$label" "$destination" >&2
        exit 1
    fi
    if [[ -d "$destination/.git" || -f "$destination/.git" ]]; then
        if [[ -n "$revision" ]]; then
            git -C "$destination" fetch --tags --quiet
            git -C "$destination" checkout --quiet "$revision"
        fi
        printf 'using existing %s checkout: %s\n' "$label" "$destination"
        return
    fi
    if [[ -d "$destination" ]] && [[ -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf 'error: %s exists and is not an empty Git checkout: %s\n' "$label" "$destination" >&2
        printf 'move it aside or set the repository layout explicitly; no files were overwritten\n' >&2
        exit 1
    fi
    if [[ -z "$url" ]]; then
        printf 'error: repository URL for %s is not configured\n' "$label" >&2
        printf 'set the corresponding WHG_*_REPO_URL environment variable\n' >&2
        exit 1
    fi
    mkdir -p "$(dirname "$destination")"
    git clone "$url" "$destination"
    if [[ -n "$revision" ]]; then
        git -C "$destination" checkout --quiet "$revision"
    fi
    printf 'cloned %s: %s\n' "$label" "$destination"
}

mkdir -p "$THIRD_PARTY_ROOT/mano/assets/mano"
clone_or_check "$THIRD_PARTY_ROOT/hamer" "$HAMER_REPO_URL" "$HAMER_REPO_REF" "HaMeR"
clone_or_check "$THIRD_PARTY_ROOT/hamer/third-party/ViTPose" "$VITPOSE_REPO_URL" "$VITPOSE_REPO_REF" "ViTPose"
clone_or_check "$THIRD_PARTY_ROOT/vggt-omega" "$VGGTO_REPO_URL" "$VGGTO_REPO_REF" "VGGT-Omega"

cat <<EOF

Third-party source layout is ready under:
  $THIRD_PARTY_ROOT

MANO model files are licensed assets and must be placed manually at:
  $THIRD_PARTY_ROOT/mano/assets/mano/MANO_LEFT.pkl
  $THIRD_PARTY_ROOT/mano/assets/mano/MANO_RIGHT.pkl

The model checkpoints, detector weights, and ViTPose weights are also not
fetched automatically. Set WHG_* paths or place them at the locations printed
by scripts/check_environment.sh before running training or evaluation.
EOF
