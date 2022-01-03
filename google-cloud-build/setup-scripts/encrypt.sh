#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
ENV_FILE=".env/env.yaml"
KEY_RING_NAME=$(yq -r .key_ring_name < "$ENV_FILE")
KEY_NAME=$(yq -r .key_name < "$ENV_FILE")
read -s -r SECRET_DECRYPTED
echo -n "$SECRET_DECRYPTED" | gcloud kms encrypt --plaintext-file=- --ciphertext-file=- --location=global --keyring="$KEY_RING_NAME" --key="$KEY_NAME" | base64 -w 0
