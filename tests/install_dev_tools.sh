#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

apt-get -y update && \
    apt-get install -y vim iproute2 tmux && \
    apt-get -y clean && \
    apt-get -y autoremove