#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
REBUILD=$2
FLAVOR=$1
ADDITIONAL_ARGUMENTS=""
if [ "$REBUILD" == "True" ]
then
  ADDITIONAL_ARGUMENTS="--force-rebuild"
fi
BUILD_PARAMETER="--no-shortcut-build"
SYSTEM_PARAMETER="--workers 7"
if [ -f /workspace/build-status.txt ]
then
  rm /workspace/build-status.txt
fi
touch /workspace/build-status.txt
COMMAND="./exaslct build --flavor-path "flavors/$FLAVOR" $BUILD_PARAMETER $ADDITIONAL_ARGUMENTS $SYSTEM_PARAMETER"
echo "Executing Command: $COMMAND"
$COMMAND || echo "fail" >> /workspace/build-status.txt
echo
COMMAND="./exaslct build-test-container $ADDITIONAL_ARGUMENTS $SYSTEM_PARAMETER"
echo "Executing Command: $COMMAND"
$COMMAND || echo "fail" >> /workspace/build-status.txt

echo "/workspace/build-status.txt"
cat /workspace/build-status.txt 

echo
echo "=========================================================="
echo "Printing docker images"
echo "=========================================================="
echo
docker images | grep exa

# TODO use internal cache without commit hash as source and produce release target except for master branch builds
