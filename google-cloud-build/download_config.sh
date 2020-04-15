#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

env_file=".env/env.yaml"
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PROJECT=$1
CONFIG_BUCKET_NAME=$2
CONFIG_BUCKET="gs://$CONFIG_BUCKET_NAME/*"
gcloud config set project "$PROJECT"
gsutil -m cp -r "$CONFIG_BUCKET" "file://.env" 
