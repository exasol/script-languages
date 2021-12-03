#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "You must provide output path as argument"
    exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

output_path=$1

echo Start oysteR...
Rscript "$SCRIPT_DIR/exec_oyster.R" "$output_path" &> "$output_path/oyster_debug.log" && echo "No vulnerabilities found!" || (cat "$output_path/oyster_debug.log" && exit 1)
