#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
echo "Changing to script directory  $SCRIPT_DIR"
cd "$SCRIPT_DIR" || return 1
export LIBPYEXADATAFRAME_DIR="$SCRIPT_DIR/base/python/python3"
export LD_LIBRARY_PATH="/opt/conda/cuda-compat/:$LD_LIBRARY_PATH" #Temporary hack for the Cuda ML flavor(s)
