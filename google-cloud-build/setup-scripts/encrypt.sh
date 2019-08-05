#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
ENV_FILE=".env/env.yaml"
KEY_RING_NAME=$(cat "$ENV_FILE" | yq -r .key_ring_name)
KEY_NAME=$(cat "$ENV_FILE" | yq -r .key_name)
read -s SECRET_DECRYPTED
echo -n "$SECRET_DECRYPTED" | gcloud kms encrypt --plaintext-file=- --ciphertext-file=- --location=global --keyring=$KEY_RING_NAME --key=$KEY_NAME | base64 -w 0
