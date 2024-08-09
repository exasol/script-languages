#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
echo "Changing to script directory  $SCRIPT_DIR"
cd "$SCRIPT_DIR" || return 1
export LIBEXAUDFLIB_PATH="$SCRIPT_DIR/base/libexaudflib_complete.so"
export LIBPYEXADATAFRAME_DIR="$SCRIPT_DIR/python/python3"
