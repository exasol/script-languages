#!/usr/bin/env bash

#
# this script implements packaging interface of the working copy
# it must implement at least three methods:
#
# * tar         - stream the installation directory (DIR)
#                 of the working copy to stdout.
#                 The script must be called from the parent directory
#                 of the installation directory
#                 (used by `c4 upload`)
#
# * extract     - extract fetched archive to the installation directory
#                 (used by `c4 fetch`)
#
# * methods     - show the list of supported methods (one method per line)

case "$1" in
  extract)
    mkdir -p install/release
    for i in install/*.tar.gz; do
      OUTPUT_DIR=install/release
      tar xf "$i" -C "$OUTPUT_DIR"
    done
    ;;
  tar)
    gunzip --stdout "$(ls *.tar.gz | head -n 1)"
    ;;
  methods)
    echo "extract methods tar" | xargs -n1 echo
    ;;
  *)
    echo "Usage: $0 extract|create|methods"
    exit 1
    ;;
esac

