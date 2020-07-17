#!/bin/bash

set -e
set -u
set -o pipefail

PACKAGE_LIST_FILE=$1
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

PACKAGE_LIST=$($SCRIPT_DIR/generate_package_list_without_version.pl --file $PACKAGE_LIST_FILE)
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
