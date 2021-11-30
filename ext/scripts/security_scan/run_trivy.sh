#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "You must provide output path as argument"
    exit 1
fi

output_path=$1

trivy rootfs --format json / >> "$output_path/trivy_report.json"
trivy rootfs --format table /
