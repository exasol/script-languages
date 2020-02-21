#!/bin/bash
set -o errexit
set -o pipefail
CLIENT_BINARY=$1
CLIENT_WRAPPER=$2
WRAPPER_TEMPLATE=$3
touch "$CLIENT_WRAPPER"
cat "$WRAPPER_TEMPLATE" >> "$CLIENT_WRAPPER"
echo >> "$CLIENT_WRAPPER"
echo 'NAME="$1"' >> "$CLIENT_WRAPPER"
echo 'OUTPUT_FILE="/tmp/$(echo $NAME | sed s#[/:]##g)"' >> "$CLIENT_WRAPPER"
echo ./$(basename "$CLIENT_BINARY") '$*' '&> "$OUTPUT_FILE"' >>  "$CLIENT_WRAPPER"
echo 'UPLOAD="curl --fail -X PUT -T $OUTPUT_FILE http://w:write@localhost:6583/default$OUTPUT_FILE"' >>  "$CLIENT_WRAPPER"
echo 'echo $UPLOAD' >>  "$CLIENT_WRAPPER"
echo '$UPLOAD' >>  "$CLIENT_WRAPPER"

