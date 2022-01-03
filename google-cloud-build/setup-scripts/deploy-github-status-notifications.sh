#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
echo "Setup githubBuildStatusNotification"
"$SCRIPT_DIR/create_encrypted_github_token.sh"
env_file=".env/env.yaml"
encrypted_github_token_file=".env/encrypted_github_token.yaml"
ENCRYPTED_GITHUB_TOKEN=$(yq -r .github_token < "$encrypted_github_token_file")
KEY_RING_NAME=$(yq -r .key_ring_name < "$env_file")
KEY_NAME=$(yq -r .key_name < "$env_file")

PROJECT_NAME=$(yq -r .gcloud_project_name < "$env_file")
cd github-status-notifications 
gcloud functions deploy githubBuildStatusNotification --runtime nodejs8 --trigger-topic cloud-builds --set-env-vars ENCRYPTED_GITHUB_TOKEN="$ENCRYPTED_GITHUB_TOKEN" --set-env-vars KEY_RING_NAME="$KEY_RING_NAME" --set-env-vars KEY_NAME="$KEY_NAME" --service-account "build-cloud-functions@$PROJECT_NAME.iam.gserviceaccount.com"
echo "Done with setup of the githubBuildStatusNotification"

