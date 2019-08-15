#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
OUTPUT_FILE=".env/encrypted_docker_password.yaml"
if [ ! -f "$OUTPUT_FILE" ]
then
        echo "Enter the Docker Passowrd"
        echo "docker_password: \"$($SCRIPT_DIR/encrypt.sh)\"" > "$OUTPUT_FILE"
fi
