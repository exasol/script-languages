#!/bin/bash
set -o errexit
set -o nounset
FLAVOR=$1
touch /workspace/build-status.txt
./exaslct run-db-test --flavor-path "flavors/$FLAVOR"  --workers 7 || echo "fail" > /workspace/build-status.txt
