#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
OUTPUT_FILE=".env/encrypted_github_token.yaml"
if [ ! -f "$OUTPUT_FILE" ]
then
        echo "Enter the Github Token"
        echo "github_token: \"$($SCRIPT_DIR/encrypt.sh)\"" > "$OUTPUT_FILE"
fi
