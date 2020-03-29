#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

grep --only-matching --exclude=error_codes.yml --exclude-dir=bazel*  --exclude-dir=.build_output --exclude-dir=script-languages -I  --line-number -R -E "[FEW]-[A-Z]+(\.[A-Z]+)*-[0-9]+" | sed -E "s/^(.+:[0-9]+):(.*)/\1\t\2/g" | sort --field-separator "-" --key=2,2 --key=3,3n 
