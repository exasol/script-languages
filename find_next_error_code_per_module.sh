#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

error_codes=$(bash find_error_codes.sh | cut -f 2)
highest_error_codes_per_module=$(bash find_highest_error_codes_per_module.sh)

for highest_error_code_for_module in $highest_error_codes_per_module
do
  module=$(echo $highest_error_code_for_module | cut -f 1 -d ",")
  highest_error_code=$(echo $highest_error_code_for_module | cut -f 2 -d ",")
#  echo module: $module
#  echo highest_error_code: $highest_error_code
  error_codes_for_module=$(echo $error_codes | tr " " "\n" | cut -f 2,3 -d "-" | grep "$module" | cut -f 2)
  miss_algined_error_code=$(echo "$error_codes_for_module" | tr " " "\n" | cut -f 2 -d "-" | awk 'BEGIN{ line=1 } { if ($1 != line) { print $1; exit } ; line++}')
  if [ -n "$miss_algined_error_code" ]
  then
    echo $module-$((miss_algined_error_code-1))
  else
    echo $module-$((highest_error_code+1))
  fi
done
