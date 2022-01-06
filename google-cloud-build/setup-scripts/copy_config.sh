#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

env_file=".env/env.yaml"
CONFIG_BUCKET=$(yq -r .config_bucket < "$env_file")
gsutil -m rsync "$@" -r "file://.env" "$CONFIG_BUCKET"
