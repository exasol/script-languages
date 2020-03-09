#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

env_file=".env/env.yaml"
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PROJECT=$(cat "$env_file" | yq -r .gcloud_project_name)
gcloud config set project "$PROJECT"
$SCRIPT_DIR/setup-scripts/update_triggers.sh
$SCRIPT_DIR/setup-scripts/copy_config.sh
