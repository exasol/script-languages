#!/bin/bash
set -o errexit
set -o pipefail
CLIENT_BINARY=$1
CLIENT_WRAPPER=$2
WRAPPER_TEMPLATE=$3
touch "$CLIENT_WRAPPER"

{
cat "$WRAPPER_TEMPLATE"
echo
echo ./"$(basename "$CLIENT_BINARY")" '$*'
} >> "$CLIENT_WRAPPER"