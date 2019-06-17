#!/usr/bin/env bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
command="./exaslct push --force-push --push-all --build-name '$($SCRIPT_DIR/build_name.sh)' $($SCRIPT_DIR/docker_options_deploy_commit.sh) $*"
echo "Executing command: $command"
bash -c "$command"