#!/usr/bin/env bash

set -euo pipefail

if [ -z "${1-}" ]; then
  VERSION="$(git rev-parse HEAD 2>/dev/null || echo latest)"
else
  VERSION="$1"
fi

echo "exasol/script-language-container:container-tool-runner-$VERSION"
