#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

echo "Setup githubBuildStatusNotification"
setup_scripts=setup-scripts
triggers=triggers
$setup_scripts/create_encrypted_github_token.sh
env_file=".env/env.yaml"
encrypted_github_token_file=".env/encrypted_github_token.yaml"
ENCRYPTED_GITHUB_TOKEN=$(cat "$encrypted_github_token_file" | yq -r .github_token)
KEY_RING_NAME=$(cat "$env_file" | yq -r .key_ring_name)
KEY_NAME=$(cat "$env_file" | yq -r .key_name)

PROJECT_NAME=$(cat "$env_file" | yq -r .gcloud_project_name)
cd github-status-notifications 
gcloud functions deploy githubBuildStatusNotification --runtime nodejs8 --trigger-topic cloud-builds --set-env-vars ENCRYPTED_GITHUB_TOKEN=$ENCRYPTED_GITHUB_TOKEN --set-env-vars KEY_RING_NAME=$KEY_RING_NAME --set-env-vars KEY_NAME=$KEY_NAME --service-account build-cloud-functions@$PROJECT_NAME.iam.gserviceaccount.com
echo "Done with setup of the githubBuildStatusNotification"

