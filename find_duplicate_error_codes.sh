#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

DUPLICATES=$(bash find_error_codes.sh | uniq -D -f 2 | awk '{print $1 "\t" $2 "-" $3 "-" $4}')
if [ -z "$DUPLICATES" ]
then
  echo "No duplicated error codes found"
  exit 0
else
  echo "Duplicated error codes found"
  echo
  echo "$DUPLICATES"
  exit 1
fi
