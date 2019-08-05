#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

env_file=".env/env.yaml"
PROJECT=$(cat "$env_file" | yq -r .gcloud_project_name)
gcloud config set project "$PROJECT"
./setup-scripts/create_encryption_key.sh
./setup-scripts/deploy-github-status-notifications.sh
./setup-scripts/update_triggers.sh
