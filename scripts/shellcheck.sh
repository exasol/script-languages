#!/bin/bash

set -u

interesting_paths=("scripts" "emulator" "exaudfclient" "ext" "githooks" "google-cloud-build")

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
status=0

for path in "${interesting_paths[@]}"; do
  find "$SCRIPT_DIR/../$path" -name '*.sh' -type f -print0 | xargs -0 -n1 shellcheck -x
  test $? -ne 0 && status=1
done


interesting_flat_paths=(".")

for flat_path in "${interesting_flat_paths[@]}"; do
  find "$SCRIPT_DIR/../$flat_path" -maxdepth 1 -name '*.sh' -type f -print0 | xargs -0 -n1 shellcheck -x
  test $? -ne 0 && status=1
done

exit $status
