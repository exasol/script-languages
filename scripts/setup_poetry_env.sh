#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

PYTHON_VERSION=$1

PYTHON_BIN=$(command -v "$PYTHON_VERSION")
poetry env use "$PYTHON_BIN"
poetry install