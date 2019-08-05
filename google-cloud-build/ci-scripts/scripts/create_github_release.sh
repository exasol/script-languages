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
find .build_output/exports -type f -size +1900M | xargs rm
EXPORTED_CONTAINERS=.build_output/exports/*.tar.gz
GITHUB_USER="$3"
GITHUB_TOKEN="$(cat secrets/GITHUB_TOKEN)"
github-release "$TAG_NAME" $EXPORTED_CONTAINERS --commit $COMMIT \
                                     --tag "$TAG_NAME" \
                                     --prerelease \
                                     --github-repository "$GITHUB_USER/script-languages" \
                                     --github-access-token "$GITHUB_TOKEN"
