#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

grep -I  --line-number -R -E "[FEW]-[A-Z]+(\.[A-Z]+)*-[0-9]+" | sed -E "s/^(.*:[0-9]+):.*([FEW]-[A-Z]+(\.[A-Z]+)*-[0-9]+).*$/\1\t\2/g"
