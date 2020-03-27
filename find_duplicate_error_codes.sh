

#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

DUPLICATES=$(bash find_error_codes.sh | cut -f 1,2,3 -d "-" --output-delimiter " " | sort -k2 -k 3n | uniq -D -f 1 | cut -f 1,2,3 -d " " --output-delimiter "-")
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
