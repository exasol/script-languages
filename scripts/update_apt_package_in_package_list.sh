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

FILES=$(grep -E -R "^$PACKAGE\|$CURRENT_VERSION" "$SEARCH_DIRECTORY" | cut -f 1 -d ":")
for FILE in $FILES
do
  echo "Found package $PACKAGE|$CURRENT_VERSION in $FILE"
  echo "Original lines:"
  grep -E "^$PACKAGE\|$CURRENT_VERSION" "$FILE"
  echo "Updated lines:"
  SEARCH_REPLACE_PATTERN="s/^($PACKAGE\|$CURRENT_VERSION).*$/$PACKAGE|$CANDIDATE_VERSION/g"
	sed -E "$SEARCH_REPLACE_PATTERN" "$FILE" | grep -E "^$PACKAGE\|"
  if [[ "$REPLACE" == "yes"  ]]
  then
    echo "Updating file $FILE:"
	  sed -i -E "$SEARCH_REPLACE_PATTERN" "$FILE"
  fi
  echo
done
