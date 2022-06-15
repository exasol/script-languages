#!/bin/bash

set -e
set -u
set -o pipefail

# Install micromamba
curl -L "https://micromamba.snakepit.net/api/micromamba/linux-64/$1" | tar -xvj "bin/micromamba"

# Setup MAMBA_ROOT_PREFIX
mkdir -p "$MAMBA_ROOT_PREFIX/conda-meta" && \
chmod -R a+rwx "$MAMBA_ROOT_PREFIX" && \

# Activate micromamba for the bash
echo "source /usr/local/bin/_activate_current_env.sh" >> ~/.bashrc && \
echo "source /usr/local/bin/_activate_current_env.sh" >> /etc/skel/.bashrc && \
ln -s /usr/local/bin/_activate_current_env.sh /etc/profile.d/_activate_current_env.sh && \
chmod -R a+rx /usr/local/bin/_activate_current_env.sh

# Add conda lib directory to ld.so.conf
echo "$MAMBA_ROOT_PREFIX/lib" > /etc/ld.so.conf.d/conda.conf
