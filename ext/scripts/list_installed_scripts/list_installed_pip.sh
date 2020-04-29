#!/bin/bash

set -e
set -u
set -o pipefail

if [[ "$1" =~ .*python.* ]]
then
  $1 -m pip list --format columns | tail -n +3 | sed "s/  */|/" | sort -f -d
else
  echo "'$1' not a python binary"
  exit 1
fi
