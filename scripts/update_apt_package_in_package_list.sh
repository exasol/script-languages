#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

if [ $# -eq 0 ];
then
  echo '"Package|Installed|Candidate" SEARCH_DIRECTORY REPLACE'
  exit 1
fi

LIST_NEWEST_VERSION_OUTPUT=$1 # Package|Installed|Candidate
SEARCH_DIRECTORY=$2
REPLACE=$3

PACKAGE=$(echo "$LIST_NEWEST_VERSION_OUTPUT" | cut -f 1 -d "|")
CANDIDATE_VERSION=$(echo "$LIST_NEWEST_VERSION_OUTPUT" | cut -f 3 -d "|")
CURRENT_VERSION=$(echo "$LIST_NEWEST_VERSION_OUTPUT" | cut -f 2 -d "|")

FILES=$(grep -R "^$PACKAGE|$CURRENT_VERSION" "$SEARCH_DIRECTORY" | cut -f 1 -d ":")
for FILE in $FILES
do
  echo "Found package $PACKAGE|$CURRENT_VERSION in $FILE"
  echo "Original lines:"
  grep "^$PACKAGE|$CURRENT_VERSION" "$FILE"
  echo "Updated lines:"
  CURRENT_VERSION_ESCAPE=${CURRENT_VERSION//\~/\\~}
  SEARCH_REPLACE_PATTERN="s/^($PACKAGE\|$CURRENT_VERSION_ESCAPE).*$/$PACKAGE|$CANDIDATE_VERSION/g"
	sed -E "$SEARCH_REPLACE_PATTERN" "$FILE" | grep "^$PACKAGE|"
  if [[ "$REPLACE" == "yes"  ]]
  then
    echo "Updating file $FILE:"
	  sed -E -i "$SEARCH_REPLACE_PATTERN" "$FILE"
  fi
  echo
done
