#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
echo "Changing to script directory  $SCRIPT_DIR"
cd "$SCRIPT_DIR" || return 1
export LIBPYEXADATAFRAME_DIR="$SCRIPT_DIR/external/exaudfclient_base+/python/python3"

if [[ -d /opt/conda ]]
then
    export CONDA_DEFAULT_ENV=base
    export CONDA_PREFIX=/opt/conda
    export MAMBA_ROOT_PREFIX=$CONDA_PREFIX
    export LD_LIBRARY_PATH="$CONDA_PREFIX/cuda-compat/:$LD_LIBRARY_PATH" # Temporary hack for the Cuda ML flavor(s)
fi
