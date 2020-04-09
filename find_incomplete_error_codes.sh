#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

grep --exclude-dir=.build_output -I  --line-number -R -E "([FEW]-)?[A-Z]+(\.[A-Z]+)*-[0-9]+" | grep -v -E "[FEW]-[A-Z]+(\.[A-Z]+)*-[0-9]+"
