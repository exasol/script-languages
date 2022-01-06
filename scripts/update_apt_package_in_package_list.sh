#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

LIST_NEWEST_VERSION_OUTPUT=$1 # Package|Installed|Candidate
FLAVOR=$2
PACKAGE=$(echo "$LIST_NEWEST_VERSION_OUTPUT" | cut -f 1 -d "|")
CANDIDATE_VERSION=$(echo "$LIST_NEWEST_VERSION_OUTPUT" | cut -f 3 -d "|")
REPLACE=$3

if [[ "$REPLACE" == "yes"  ]]
then
	SED_REPLACE_OPTION="-i"
else
	SED_REPLACE_OPTION=""
fi
grep -E -R "^$PACKAGE\|" "$FLAVOR" | cut -f 1 -d ":" | xargs -n 1 sed "$SED_REPLACE_OPTION" -E "s/^($PACKAGE)\|[^# ]*( *#.*)?$/$PACKAGE|$CANDIDATE_VERSION/g" | grep -E "^$PACKAGE\|"
