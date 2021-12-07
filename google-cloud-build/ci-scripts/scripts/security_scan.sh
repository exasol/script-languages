#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
FLAVOR=$1
SYSTEM_PARAMETER="--workers 7"
if [ -f /workspace/build-status.txt ]
then
  rm /workspace/build-status.txt
fi
touch /workspace/build-status.txt
COMMAND="./exaslct security-scan --flavor-path "flavors/$FLAVOR" $SYSTEM_PARAMETER"
echo "Executing Command: $COMMAND"
$COMMAND || echo "fail" >> /workspace/build-status.txt
echo

echo "/workspace/build-status.txt"
cat /workspace/build-status.txt
echo

SECURITY_REPORT_OUTPUT_PATH=".build_output/security_scan"

echo "============= SECURITY REPORT ==========="
echo
cat "$SECURITY_REPORT_OUTPUT_PATH/security_report"