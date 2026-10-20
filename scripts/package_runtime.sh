#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$ROOT/world-hand-gen-next-runtime.zip"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    printf 'error: python3 or python is required to build the runtime archive\n' >&2
    exit 1
fi

TEMP_OUTPUT="$(mktemp "$ROOT/.world-hand-gen-next-runtime.XXXXXX.zip")"
cleanup() {
    rm -f -- "$TEMP_OUTPUT"
}
trap cleanup EXIT

"$PYTHON_BIN" - "$ROOT" "$TEMP_OUTPUT" <<'PY'
from __future__ import annotations

import fnmatch
import os
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
archive_path = Path(sys.argv[2]).resolve()

# These are the project trees that form the runtime package. Missing trees are
# intentionally skipped so the script also works with a minimal checkout.
include_dirs = ("src", "tests", "scripts", "docs")
root_metadata = (
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    ".gitignore",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "environment.yml",
    "LICENSE",
    "LICENSE.txt",
    "CITATION.cff",
)


def archive_name(path: Path) -> str:
    return path.relative_to(root).as_posix()


def forbidden(name: str) -> bool:
    """Return whether an archive path matches a runtime exclusion."""
    parts = Path(name).parts
    if not parts:
        return True
    # data/ is a repository-root input mount. src/data/ is source code and is
    # therefore deliberately allowed.
    if parts[0] == "data":
        return True
    forbidden_parts = {
        ".git",
        ".zcode",
        "__pycache__",
        ".pytest_cache",
        "cache",
        "experiments",
        "eval_results",
        "vis",
        "demo_out",
        "outputs",
        "example_data",
        "logs",
        "*.egg-info",
    }
    if any(part in forbidden_parts for part in parts):
        return True
    if any(fnmatch.fnmatch(part, pattern) for part in parts for pattern in forbidden_parts):
        return True
    if any(part.startswith("tmp_") for part in parts):
        return True
    if any(fnmatch.fnmatch(part, "*_demo_data.tar.gz") for part in parts):
        return True
    if any(
        fnmatch.fnmatch(part, "*.pyc")
        or fnmatch.fnmatch(part, "*.pyo")
        or fnmatch.fnmatch(part, "*.pyd")
        or fnmatch.fnmatch(part, "*.tmp")
        for part in parts
    ):
        return True
    return False


files: set[str] = set()
directories: set[str] = set()

for relative_dir in include_dirs:
    top = root / relative_dir
    if not top.is_dir():
        continue
    top_name = top.relative_to(root).as_posix()
    if forbidden(top_name):
        continue
    directories.add(top_name + "/")
    for directory, dirnames, filenames in os.walk(top, topdown=True, followlinks=False):
        directory_path = Path(directory)
        current_name = archive_name(directory_path)
        if forbidden(current_name):
            dirnames[:] = []
            continue
        directories.add(current_name + "/")
        kept_dirs = []
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            child_name = archive_name(child)
            if forbidden(child_name) or child.is_symlink():
                continue
            kept_dirs.append(dirname)
            directories.add(child_name + "/")
        dirnames[:] = kept_dirs
        for filename in sorted(filenames):
            child = directory_path / filename
            child_name = archive_name(child)
            if forbidden(child_name) or child.is_dir() or child.is_symlink():
                continue
            files.add(child_name)

for relative_file in root_metadata:
    path = root / relative_file
    if path.is_file() and not path.is_symlink() and not forbidden(relative_file):
        files.add(relative_file)

# Write and inspect the archive before publishing it at the requested path.
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(directories):
        archive.writestr(name, b"")
    for name in sorted(files):
        archive.write(root / name, arcname=name)

with zipfile.ZipFile(archive_path, "r") as archive:
    names = archive.namelist()
    bad = sorted(name for name in names if forbidden(name.rstrip("/")))
    # Keep this explicit check separate from forbidden() so a future change to
    # source-tree filtering cannot accidentally permit a root data mount.
    bad_data = sorted(
        name for name in names
        if name == "data" or name.startswith("data/")
    )
    if bad_data or bad:
        details = sorted(set(bad_data + bad))
        raise SystemExit("error: forbidden paths in runtime archive: " + ", ".join(details))
    if any(name == "" for name in names):
        raise SystemExit("error: runtime archive contains an empty path")
    required = {
        "src/model/flow/state.py",
        "src/model/wrappers/mano_fit.py",
    }
    missing = sorted(required - set(names))
    if missing:
        raise SystemExit("error: runtime archive is missing required MANO-flow files: " + ", ".join(missing))
    state_source = archive.read("src/model/flow/state.py").decode("utf-8")
    fit_source = archive.read("src/model/wrappers/mano_fit.py").decode("utf-8")
    if "class ManoFlowState" not in state_source or "-> ManoFlowState" not in fit_source:
        raise SystemExit("error: runtime archive does not expose mano_fit with ManoFlowState output")

print(f"created {archive_path} ({len(files)} files, {len(directories)} directories)")
PY

mv -f -- "$TEMP_OUTPUT" "$OUTPUT"
trap - EXIT
printf 'runtime archive: %s\n' "$OUTPUT"
