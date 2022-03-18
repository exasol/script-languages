#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
FLAVOR=$1
SYSTEM_PARAMETER="--workers 7"
COMMAND="python3 -m exasol_script_languages_container_tool.main security-scan --flavor-path "flavors/$FLAVOR" $SYSTEM_PARAMETER"
echo "Executing Command: $COMMAND"
$COMMAND
echo

SECURITY_REPORT_OUTPUT_PATH=".build_output/security_scan"

echo "============= SECURITY REPORT ==========="
echo
cat "$SECURITY_REPORT_OUTPUT_PATH/security_report"