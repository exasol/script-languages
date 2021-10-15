#!/usr/bin/env bash

set -u

die() { echo "ERROR:" "$@" >&2; exit 1; }

# $1: path to python exatest file
# $2...: additional arguments passed to tests
function run_test() {
  echo "Starting test: $1" 1>&2
  python3 "$@"
  rc=$?
  if [ $rc != 0 ]; then
      echo "FAILED: $1"
      exit 1
  fi
}

# $1: folder name
# $2: exasol server address
# $3: odbc_driver
function run_tests_in_folder() {
  for test_ in $1/*.py; do
      run_test "$test_" --server "$2" --driver "$3"
  done
}


optarr=$(getopt -o 'h' --long 'help,server:,exatest-folder:,odbc-driver:,' -- "$@")

eval set -- "$optarr"

# ./run-locally.sh --server 192... --exatest-folder /home/.../tests
while true; do
    case "$1" in
        --server) server="$2"; shift 2;;
        --odbc-driver) odbc_driver="$2"; shift 2;;
        --) shift; break;;
        *) echo "Usage: $0"
		       echo "Options:"
		       echo "  [--server=<host:port>]                Address of Exasol database instance"
		       echo "  [--odbc-driver=<path>]                     Path to ODBC driver"
		       echo "  [-h|--help]                           Print this help."
           echo "Environment variable EXAPLUS must point to exaplus executable."; exit 0;;
    esac
done

if [ -z "$server" ]; then die "--server is required"; fi
if [ -z "$odbc_driver" ]; then die "--odbc-driver is required"; fi

if [ -z "$EXAPLUS" ]; then
    echo "Environment variable EXAPLUS must point to exaplus executable."
    exit 1
fi


SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
run_test "$SCRIPT_DIR/test/python/network.py" --server "$server" --driver "$odbc_driver"
#run_test "$SCRIPT_DIR/test/java/unicode.py" --server "$server" --driver "$odbc_driver"
#run_test "$SCRIPT_DIR/test/python/installed_packages.py" --server "$server" --driver "$odbc_driver"
#run_tests_in_folder "$SCRIPT_DIR/test/java" "$server" "$odbc_driver"

