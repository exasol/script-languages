#!/bin/bash

set -e
set -u
set -o pipefail

PACKAGE_LIST_FILE=$1

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
echo "Package|Requested Version|Candidate Version"
"$SCRIPT_DIR/list_candidate_versions_for_apt.sh" "$PACKAGE_LIST_FILE" \
  | sed 1d \
  | awk 'BEGIN { FS = "|" } ; $2!=$3 {print $1 "|" $2 "|" $3}'
