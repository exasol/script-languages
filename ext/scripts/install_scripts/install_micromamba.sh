#!/bin/bash

set -e
set -u
set -o pipefail


curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj "bin/micromamba"
mkdir -p "$MAMBA_ROOT_PREFIX/conda-meta" && \
chmod -R a+rwx "$MAMBA_ROOT_PREFIX" && \
echo "source /usr/local/bin/_activate_current_env.sh" >> ~/.bashrc && \
echo "source /usr/local/bin/_activate_current_env.sh" >> /etc/skel/.bashrc && \
ln -s /usr/local/bin/_activate_current_env.sh /etc/profile.d/_activate_current_env.sh && \
chmod -R a+rx /usr/local/bin/_activate_current_env.sh
