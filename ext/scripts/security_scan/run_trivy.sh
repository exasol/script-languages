#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "You must provide output path as argument"
    exit 1
fi

output_path=$1

trivy rootfs --no-progress --format json --output "$output_path/trivy_report.json" /
#run with format table and print to stdout
trivy rootfs --no-progress --format table --output "$output_path/trivy_report.txt" /
#Force script to return with error if a high or critical issue is found
trivy rootfs --no-progress --severity "HIGH,CRITICAL" --exit-code 1 /