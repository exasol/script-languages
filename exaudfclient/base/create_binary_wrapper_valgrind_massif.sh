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
#shellcheck disable=SC2001
echo 'MASSIF_FILE="/tmp/$(echo "$NAME" | sed s#[/:]##g)"'
echo valgrind --tool=massif --massif-out-file='"$MASSIF_FILE"' "./$(basename "$CLIENT_BINARY")" '"$@"'
echo 'UPLOAD="curl --fail -X PUT -T $MASSIF_FILE http://w:write@localhost:6583/default$MASSIF_FILE"'
echo 'echo "$UPLOAD"'
echo '"$UPLOAD"'
} >> "$CLIENT_WRAPPER"