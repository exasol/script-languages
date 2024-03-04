#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

if [ $# -eq 0 ];
then
  echo 'SEARCH_DIRECTORY REPLACE'
  exit 1
fi

SEARCH_DIRECTORY=$1
REPLACE=$2

while read -r LIST_NEWEST_VERSION_OUTPUT; do 
    bash "$SCRIPT_DIR/update_apt_package_in_package_list.sh" "$LIST_NEWEST_VERSION_OUTPUT" "$SEARCH_DIRECTORY" "$REPLACE" || true
done
