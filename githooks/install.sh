#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
REPO_DIR=$(git rev-parse --show-toplevel)
REPO_DIR="$(readlink -f "${REPO_DIR}")"
GIT_DIR="$REPO_DIR/.git"
GIT_DIR="$(readlink -f "${GIT_DIR}")"

if [[ ! -d "$GIT_DIR" ]]; then
  if [[ -d "$REPO_DIR/../.git" ]]; then
    GIT_DIR="$REPO_DIR/../.git"
    GITHOOKS_PATH="$GIT_DIR/modules/script-languages/hooks"
  else
    echo "$GIT_DIR is not a git directory." >&2
    exit 1
  fi
else
    GITHOOKS_PATH="$GIT_DIR/hooks"
fi

GITHOOKS_PATH="$(readlink -f "${GITHOOKS_PATH}")"

copy_hook() {
    local SCRIPT_PATH="$SCRIPT_DIR/$1"
    local GITHOOK_PATH="$GITHOOKS_PATH/$2"
    local RELATIVE_PATH=$(realpath --relative-to="$GITHOOKS_PATH" "$SCRIPT_PATH")
    pushd "$GITHOOKS_PATH" > /dev/null
    if [ -e "$GITHOOK_PATH" ] || [ -L "$GITHOOK_PATH" ]
    then
      echo
      echo "Going to delete old hook $GITHOOK_PATH"
      rm "$GITHOOK_PATH" > /dev/null
    fi
    echo
    echo "Link hook to script" >&2
    echo "Hook-Path: $GITHOOK_PATH" >&2
    echo "Script-path: $SCRIPT_PATH" >&2
    echo
    ln -s "$RELATIVE_PATH" "$2" > /dev/null
    chmod +x "$SCRIPT_PATH" > /dev/null
    popd > /dev/null
}

copy_hook pre-commit pre-commit
copy_hook pre-commit post-rewrite
copy_hook pre-push pre-push
