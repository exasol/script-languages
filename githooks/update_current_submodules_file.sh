#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# define colors for use in output
green='\033[0;32m'
no_color='\033[0m'
grey='\033[0;90m'

echo -e "Checking submodules ${grey}(pre-commit hook)${no_color} "

# Jump to the current project's root directory (the one containing
# .git/)
ROOT_DIR=$(git rev-parse --show-cdup)

GITMODULES_FILE="${ROOT_DIR}.gitmodules"

if [ -f "$GITMODULES_FILE" ]
then
  SUBMODULES=$(grep path "$GITMODULES_FILE" | sed 's/^.*path = //' | sort)

  CURRENT_GITMODULES="$ROOT_DIR.current_gitmodules"
  if [ -f "$CURRENT_GITMODULES" ]
  then
    rm "$CURRENT_GITMODULES"
  fi
  for SUB in $SUBMODULES
  do
      git ls-files -s "$SUB" >> "$CURRENT_GITMODULES"
  done
  git add "$CURRENT_GITMODULES"
fi
