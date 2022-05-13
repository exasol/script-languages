#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# Utility script to update the AWS Build stack. Supposed to be called manually!

AWS_PROFILE=$1

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
ROOT_DIR="$SCRIPT_DIR/.."

pushd $ROOT_DIR > /dev/null
#poetry run python3 -m exasol_script_languages_container_ci_setup.main deploy-ci-build --aws-profile "$AWS_PROFILE" --log-level info --project ScriptLanguages --project-url "https://github.com/exasol/script-languages"
poetry run python3 -m exasol_script_languages_container_ci_setup.main deploy-release-build --aws-profile "$AWS_PROFILE" --log-level info --project ScriptLanguages --project-url "https://github.com/exasol/script-languages"

popd > /dev/null
