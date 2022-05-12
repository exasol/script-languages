#!/usr/bin/env bash

set -euo pipefail

BUILD_SPEC=$1
AWS_PROFILE=$2
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

pushd "$SCRIPT_DIR/../../" &>/dev/null
#Artifacts will be stored under /tmp
./codebuild_build.sh -e ./aws-code-build/run_local/run_local.env -d -m -c -i aws/codebuild/ubuntu/standard:5.0 -b "$BUILD_SPEC" -s "." -p $AWS_PROFILE -a /tmp
popd > /dev/null
