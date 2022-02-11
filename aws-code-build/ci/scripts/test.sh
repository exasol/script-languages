#!/bin/bash
set -o errexit
set -o nounset
FLAVOR=$1
LOG_MESSAGE=$(git log --oneline -1)
if [[ "$LOG_MESSAGE" =~ \[skip\ tests\] ]]
then
  echo "Found [skip tests] in \"$LOG_MESSAGE\". Going to skip tests."
else
  ./exaslct run-db-test --flavor-path "flavors/$FLAVOR"  --workers 7
  ./exaslct run-db-test --flavor-path "flavors/$FLAVOR"  --workers 7 --test-folder test/linker_namespace_sanity --release-goal base_test_build_run
echo
echo "=========================================================="
echo "Printing docker images"
echo "=========================================================="
echo
docker images | grep exa
fi
