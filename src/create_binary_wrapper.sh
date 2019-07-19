#!/bin/bash
set -o errexit
set -o pipefail
CLIENT_BINARY=$1
CLIENT_WRAPPER=$2
WRAPPER_TEMPLATE=$3
touch "$CLIENT_WRAPPER"
cat "$WRAPPER_TEMPLATE" >> "$CLIENT_WRAPPER"
echo >> "$CLIENT_WRAPPER"
if [ -n "$JAVA_PREFIX" ]
then
    echo 'export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:'`find "$JAVA_PREFIX" -name *.so -type f |  xargs -n 1 dirname | sort | uniq | paste -sd ":" -` >> "$CLIENT_WRAPPER"
    echo 'echo "LD_LIBRARY_PATH: $$LD_LIBRARY_PATH"' >> "$CLIENT_WRAPPER"
fi
echo ./$(basename "$CLIENT_BINARY") '$$*' >>  "$CLIENT_WRAPPER"