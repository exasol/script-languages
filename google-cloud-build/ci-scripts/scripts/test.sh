#!/bin/bash
set -o errexit
set -o nounset
FLAVOR=$1
LOG_MESSAGE=$(git log --oneline -1)
if [[ "$LOG_MESSAGE" =~ "[skip tests]" ]]
then
  echo "Found [skip tests] in \"$LOG_MESSAGE\". Going to skip tests."
else
  touch /workspace/build-status.txt
  ./exaslct run-db-test --flavor-path "flavors/$FLAVOR"  --workers 7 || echo "fail" > /workspace/build-status.txt
echo
echo "=========================================================="
echo "Printing docker images"
echo "=========================================================="
echo
docker images | grep exa
fi
