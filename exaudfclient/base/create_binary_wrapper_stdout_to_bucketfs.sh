#!/bin/bash
# shellcheck disable=SC2016

set -o errexit
set -o pipefail
CLIENT_BINARY=$1
CLIENT_WRAPPER=$2
WRAPPER_TEMPLATE=$3
touch "$CLIENT_WRAPPER"

{
cat "$WRAPPER_TEMPLATE"
echo
echo 'NAME="$1"'
#Ignore shellcheck rule, change is too risky.
#shellcheck disable=SC2001
echo 'OUTPUT_FILE="/tmp/$(echo "$NAME" | sed s#[/:]##g)"'
echo "./$(basename "$CLIENT_BINARY")" '"$@"' '&> "$OUTPUT_FILE"'
echo 'UPLOAD="curl --fail -X PUT -T $OUTPUT_FILE http://w:write@localhost:6583/default$OUTPUT_FILE"'
echo 'echo "$UPLOAD"'
echo '"$UPLOAD"'

} >> "$CLIENT_WRAPPER"