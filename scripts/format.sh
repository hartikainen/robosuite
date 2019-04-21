 #!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
builtin cd "${PROJECT_ROOT}" || exit 1

YAPF_FLAGS=(
    '--style' "${PROJECT_ROOT}/.style.yapf"
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'robosuite/models/assets/*'
)

format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" tests robosuite
}


format
