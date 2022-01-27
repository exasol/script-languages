#!/bin/bash

set -e
set -u
set -o pipefail

assert_test_failed=0

function assert() {
  cmpA=$1
  shift 1
  cmpB="${*}"
  if [[ "$DRY_RUN_OPTION" == "--dry-run" ]]; then
    if [[ $cmpA != "$cmpB" ]]; then
      >&2 echo "ERROR: '$cmpA' does not match '$cmpB'"
      assert_test_failed=1
    fi
  else
    echo "$cmpA"
  fi
}

function check_for_failed_tests() {
  if [[ $assert_test_failed -eq 1 ]]; then
    <&2 echo "Some tests failed"
    exit 1
  fi
}
