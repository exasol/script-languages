#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "You must provide output path as argument"
    exit 1
fi

output_path=$1
mkdir -p "$output_path"

#shellcheck disable=SC2153
echo "$SECURITY_SCANNERS"

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

if [ -z "${SECURITY_SCANNERS-}" ]; then
  echo "No security scanners set. Please set SECURITY_SCANNERS."
  exit 1
fi

function _run_security_scanners() {

  for current_scanner in "$@"; do

    case "$current_scanner" in
      trivy)
        "$SCRIPT_DIR/run_trivy.sh" "$output_path"
        ;;
      oyster)
        "$SCRIPT_DIR/run_oyster.sh" "$output_path"
        ;;
      *)
        echo "Unknown scanner: $current_scanner"
        exit 1
        ;;
    esac
  done
}

security_scanners=$SECURITY_SCANNERS
#shellcheck disable=SC2086
_run_security_scanners $security_scanners

