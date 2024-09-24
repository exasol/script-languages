#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "You must provide output path as argument"
    exit 1
fi

TRIVY_CACHE_LOCATION="https://dli4ip9yror05.cloudfront.net"

mkdir -p "$HOME/.cache/trivy/db" "$HOME/.cache/trivy/java-db"
curl -o  "$HOME/.cache/trivy/db/metadata.json" "${TRIVY_CACHE_LOCATION}/db/metadata.json"
curl -o  "$HOME/.cache/trivy/db/trivy.db" "${TRIVY_CACHE_LOCATION}/db/trivy.db"
curl -o  "$HOME/.cache/trivy/java-db/metadata.json" "${TRIVY_CACHE_LOCATION}/java-db/metadata.json"
curl -o  "$HOME/.cache/trivy/java-db/trivy-java.db" "${TRIVY_CACHE_LOCATION}/java-db/trivy-java.db"



output_path=$1

trivy rootfs  --no-progress --offline-scan --format json --skip-db-update --ignore-policy /trivy.rego --output "$output_path/trivy_report.json" / > /dev/null
#run with format table and print to stdout
trivy rootfs --no-progress --offline-scan --format table --skip-db-update --ignore-policy /trivy.rego --output "$output_path/trivy_report.txt" / > /dev/null
#Force script to return with error if a high or critical issue is found
trivy rootfs --no-progress --offline-scan --skip-db-update --ignore-policy /trivy.rego --show-suppressed --severity "HIGH,CRITICAL" --exit-code 1 /