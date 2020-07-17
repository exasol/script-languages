#!/bin/bash

set -e
set -u
set -o pipefail

PACKAGE_LIST_FILE=$1


./list_candidate_versions_for_apt.sh $PACKAGE_LIST_FILE \
  | sed 1d \
  | tr "|" " " \
  | awk '$2!=$3 {print $1 " " $2 " " $3}' \
  | tr " " "|" \
  | tr "\n" " " \
  | sed "s/^/Package|Installed|Candidate /g" \
  | tr " " "\n" 
