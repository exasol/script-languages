#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "You must provide output path as argument"
    exit 1
fi

output_path=$1

trivy rootfs --no-progress --offline-scan --format json --ignore-policy /trivy.rego --output "$output_path/trivy_report.json" / > /dev/null
#run with format table and print to stdout
trivy rootfs --no-progress --offline-scan --format table --ignore-policy /trivy.rego --output "$output_path/trivy_report.txt" / > /dev/null
#Force script to return with error if a high or critical issue is found
trivy rootfs --no-progress --offline-scan --ignore-policy /trivy.rego --show-suppressed --severity "HIGH,CRITICAL" --exit-code 1 /