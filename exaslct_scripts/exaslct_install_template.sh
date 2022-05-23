#!/usr/bin/env bash

#####################################################################################
###REMEMBER TO TEST ANY CHANGES HERE ON MACOSX!!!
#####################################################################################

set -euo pipefail

rl=readlink
if [[ "$(uname)" = Darwin ]]; then
  rl=greadlink
fi

if [[ ! "$(command -v $rl)" ]]; then
  echo readlink not available! Please install coreutils: On Linux \"apt-get install coreutils\" or similar. On MacOsX \"brew install coreutils\".
  exit 1
fi

SCRIPT_DIR="$(dirname "$($rl -f "${BASH_SOURCE[0]}")")"

EXASLCT_GIT_REF="<<<<EXASLCT_GIT_REF>>>>"
RUNNER_IMAGE_NAME="$(bash "$SCRIPT_DIR/construct_docker_runner_image_name.sh" "$EXASLCT_GIT_REF")"

bash "$SCRIPT_DIR/exaslct_within_docker_container_without_container_build.sh" "$RUNNER_IMAGE_NAME" "${@}"

