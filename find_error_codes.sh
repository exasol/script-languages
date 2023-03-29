#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

grep --only-matching \
     --exclude=error_codes.yml \
     --exclude-dir=bazel* \
     --exclude-dir=.build_output \
     --exclude-dir=script-languages \
     --exclude-dir=test_container \
     -I  --line-number -R -E \
  "[FEW]-[A-Z]+(-[A-Z]+)*-[0-9]+" \
  | sed -E "s/^(.+:[0-9]+):(.*)$/\1\t\2/g" \
  | sed -E "s/^(.*)\t([FEW])-([A-Z]+(-[A-Z]+)*)-([0-9]+)$/\1\t\2\t\3\t\5/g" \
  | sort --key=3,3 --key=4,4n 
