#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

if [ $# == 0 ]; then
    echo '"Package|Installed|Candidate" [SEARCH_DIRECTORY] [REPLACE]'
    echo '- SEARCH_DIRECTORY: default .'
    echo '- REPLACE: either yes or no, default: no'
    exit 1
fi

# Format of $1: Package|Installed|Candidate
# Set array variable SPEC.
IFS='|' read -ra SPEC <<< "$1"
SEARCH_DIRECTORY=${2:-.}
REPLACE=${3:-no}

PACKAGE=${SPEC[0]}
CURRENT_VERSION=${SPEC[1]}
CANDIDATE_VERSION=${SPEC[2]}

FILES=$(grep -R "^$PACKAGE|$CURRENT_VERSION" "$SEARCH_DIRECTORY" | cut -f 1 -d ":")
for FILE in $FILES; do
    echo "Found package $PACKAGE|$CURRENT_VERSION in $FILE"
    echo "Original lines:"
    grep "^$PACKAGE|$CURRENT_VERSION" "$FILE"
    echo "Updated lines:"
    CURRENT_VERSION_ESCAPE=${CURRENT_VERSION//\~/\\~}
    SEARCH_REPLACE_PATTERN="s/^($PACKAGE\|$CURRENT_VERSION_ESCAPE).*$/$PACKAGE|$CANDIDATE_VERSION/g"
    sed -E "$SEARCH_REPLACE_PATTERN" "$FILE" | grep "^$PACKAGE|"
    if [ "$REPLACE" == "yes" ]; then
        echo "Updating file $FILE:"
        sed -E -i "$SEARCH_REPLACE_PATTERN" "$FILE"
    fi
    echo
done
