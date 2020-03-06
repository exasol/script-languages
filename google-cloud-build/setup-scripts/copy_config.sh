#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

env_file=".env/env.yaml"
CONFIG_BUCKET=$(cat "$env_file" | yq -r .config_bucket)
gsutil -m cp -r "file://.env" "$CONFIG_BUCKET"
