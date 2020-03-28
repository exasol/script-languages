#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

grep --exclude-dir=.build_output -I  --line-number -R -E "[FEW]-[A-Z]+(\.[A-Z]+)*-[0-9]+" | sed -E "s/^(.*:[0-9]+):.*([FEW]-[A-Z]+(\.[A-Z]+)*-[0-9]+).*$/\1\t\2/g" | sort --field-separator "-" --key=2,2 --key=3,3n 
