#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
TAG_NAME="$1"
COMMIT="$2"
if [ -z "$TAG_NAME" ]
then
  TAG_NAME="ci/${COMMIT:0:10}"
fi
BUILD_DIR=.build_output/cache/exports
find "$BUILD_DIR" -type f -print0 -size +1900M | xargs -0 rm
EXPORTED_CONTAINERS="$BUILD_DIR/*.tar.gz*"
GITHUB_USER="$3"
GITHUB_TOKEN="$(cat secrets/GITHUB_TOKEN)"
GITHUB_REPOSITORY="$4"
github-release "$TAG_NAME" "$EXPORTED_CONTAINERS" --commit "$COMMIT" \
                                     --tag "$TAG_NAME" \
                                     --draft \
                                     --github-repository "$GITHUB_USER/$GITHUB_REPOSITORY" \
                                     --github-access-token "$GITHUB_TOKEN"
