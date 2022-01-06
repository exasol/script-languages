#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

LIST_NEWEST_VERSION_OUTPUT=$1 # Package|Installed|Candidate
FLAVOR=$2
PACKAGE=$(echo "$LIST_NEWEST_VERSION_OUTPUT" | cut -f 1 -d "|")

grep -E -R "^$PACKAGE\|" "$FLAVOR"
