#!/bin/bash

set -e
set -u
set -o pipefail

PACKAGE_LIST_FILE=$1
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

PACKAGE_LIST=( $($SCRIPT_DIR/extract_columns_from_package_lisl.pl --file $PACKAGE_LIST_FILE --columns 0) )
VERSION_LIST=( $($SCRIPT_DIR/extract_columns_from_package_lisl.pl --file $PACKAGE_LIST_FILE --columns 1) )
echo "Package|Requested Version|Candidate Version"
while (( ${#PACKAGE_LIST[@]} ))
do
  package="${PACKAGE_LIST[0]}"
  version="${VERSION_LIST[0]}"
  if [ "$version" == "<<<<1>>>>" ]
  then
    version="No version specified"
  fi
  set +e
  candidate=$( apt-cache policy "$package" | grep "Candidate" | sed -E "s/ +//g" | cut -f2 -d ":")
  set -e
  if [ -z "$candidate" ]
  then
    candidate="Package not available"
  fi

  echo "$package|$version|$candidate"
  PACKAGE_LIST=( "${PACKAGE_LIST[@]:1}" )
  VERSION_LIST=( "${VERSION_LIST[@]:1}" )
done
