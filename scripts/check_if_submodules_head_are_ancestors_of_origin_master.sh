#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

REPO_DIR=$(git rev-parse --show-toplevel)
REPO_DIR="$(readlink -f "${REPO_DIR}")"

EXIT_CODE=0
submodule_paths=$(git submodule status --recursive | sed "s/^\s*//" | cut -f 2 -d " ")
for relative_submodule_path in $submodule_paths
do
    submodule_path="$REPO_DIR/$relative_submodule_path"
    submodule_path="$(readlink -f "${submodule_path}")"
    pushd "$submodule_path" &> /dev/null
    git fetch
    echo
    echo "Checking submodule: \"$relative_submodule_path\""
      #HASH_ORIGIN_MASTER=$(git log --decorate=full --format="%H" refs/remotes/origin/master | sed "s/, /,/g")
    HEAD_IS_ANCESTOR_OF_ORIGIN_MASTER=$(git merge-base --is-ancestor HEAD refs/remotes/origin/master || echo "$?")
    HASH_ORIGIN_MASTER=$(git log --decorate=full --format="%H" -n 1 refs/remotes/origin/master)
    HASH_HEAD=$(git log --decorate=full --format="%H" -n 1 HEAD)
    if [[ "${HEAD_IS_ANCESTOR_OF_ORIGIN_MASTER}" == "" ]]
    then
      echo "HEAD is ancestor of origin/master"
      echo "Hash HEAD:          $HASH_HEAD"
      echo "Hash origin/master: $HASH_ORIGIN_MASTER"
    else
      echo "HEAD is not ancestor of origin/master"
      echo "Hash HEAD:          $HASH_HEAD"
      echo "Hash origin/master: $HASH_ORIGIN_MASTER"
      echo "Log Line HEAD:"
      git log --oneline -n 1 HEAD
      echo "Log Line origin/master:"
      git log --oneline -n 1 refs/remotes/origin/master
      EXIT_CODE=1
    fi
    popd &> /dev/null
done
exit $EXIT_CODE
