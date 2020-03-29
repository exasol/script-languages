#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

error_codes=$(bash find_error_codes.sh | cut -f 2 | uniq) # uniq is needed in case of duplicates, because the awk to find the miss_algined_error_code fails on duplicates
highest_error_codes_per_module=$(bash find_highest_error_codes_per_module.sh)

for highest_error_code_for_module in $highest_error_codes_per_module
do
  module=$(echo $highest_error_code_for_module | cut -f 1 -d ",")
  highest_error_code=$(echo $highest_error_code_for_module | cut -f 2 -d ",")
#  echo module: $module
#  echo highest_error_code: $highest_error_code
  error_codes_for_module=$(echo $error_codes | tr " " "\n" | cut -f 2,3 -d "-" | grep "$module" | cut -f 2)
  miss_algined_error_code=$(echo "$error_codes_for_module" | tr " " "\n" | cut -f 2 -d "-" | awk 'BEGIN{ line=1000 } { if ($1 != line) { print $1; exit } ; line++}') # we assume find_error_codes.sh returns the error codes sorted by their module and number
  # echo "$error_codes_for_module" | tr " " "\n" | cut -f 2 -d "-" | awk 'BEGIN{ line=1 } { print $1 "," line ; line++}' # DEBUG Allgnment
  if [ -n "$miss_algined_error_code" ]
  then
    echo $module-$((miss_algined_error_code-1))
  else
    echo $module-$((highest_error_code+1))
  fi
done
