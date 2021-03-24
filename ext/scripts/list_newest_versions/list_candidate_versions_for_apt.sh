#!/bin/bash

set -e
set -u
set -o pipefail

PACKAGE_LIST_FILE=$1
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

PACKAGE_LIST=$($SCRIPT_DIR/extract_columns_from_package_lisl.pl --file $PACKAGE_LIST_FILE --columns 0)
apt-cache policy $PACKAGE_LIST \
  | grep -A 2 -E "^[^ ]+:" \
  | sed "s/^--/|/g" \
  | tr "\n" " " \
  | tr "|" "\n" \
  | sed "s/Candidate//g" \
  | sed "s/Installed//g" \
  | sed "s/: / /g" \
  | sed "s/^ //g" \
  | sed "s/ $//g" \
  | sed -E "s/ +/ /g" \
  | tr " " "|" \
  | tr "\n" " " \
  | sed "s/^/Package|Installed|Candidate /g" \
  | sed "s/$/ /g" \
  | tr " " "\n" 
