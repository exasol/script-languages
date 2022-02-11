#!/usr/bin/env bash

set -euo pipefail

BUILD_SPEC=$1
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

pushd "$SCRIPT_DIR/../../" &>/dev/null
./codebuild_build.sh -e ./aws-code-build/run_local/run_local.env -c -i aws/codebuild/ubuntu/standard:5.0 -b "$BUILD_SPEC" -a /tmp
popd > /dev/null