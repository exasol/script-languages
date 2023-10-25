#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

LIST_NEWEST_VERSION_OUTPUT=$1 # Package|Installed|Candidate
SEARCH_DIRECTORY=$2
REPLACE=$3

PACKAGE=$(echo "$LIST_NEWEST_VERSION_OUTPUT" | cut -f 1 -d "|")
CANDIDATE_VERSION=$(echo "$LIST_NEWEST_VERSION_OUTPUT" | cut -f 3 -d "|")
CURRENT_VERSION=$(echo "$LIST_NEWEST_VERSION_OUTPUT" | cut -f 2 -d "|")

if [[ "$REPLACE" == "yes"  ]]
then
	SED_REPLACE_OPTION=("-i")
else
	SED_REPLACE_OPTION=()
fi
grep -E -R "^$PACKAGE\|$CURRENT_VERSION" "$SEARCH_DIRECTORY" \
	| cut -f 1 -d ":" \
	| xargs -I{} sed "${SED_REPLACE_OPTION[@]}" -E "s/^($PACKAGE\|$CURRENT_VERSION).*$/$PACKAGE|$CANDIDATE_VERSION/g" "{}" \
	| grep -E "^$PACKAGE\|"
