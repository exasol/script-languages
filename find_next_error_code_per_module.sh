#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

error_codes=$(bash find_error_codes.sh | cut -f 2,3,4 | uniq |  tr '\t' ";") # uniq is needed in case of duplicates, because the awk to find the miss_algined_error_code fails on duplicates
highest_error_codes_per_module=$(bash find_highest_error_codes_per_module.sh | tr '\t' ";")

for highest_error_code_for_module in $highest_error_codes_per_module
do
  module=$(echo $highest_error_code_for_module | cut -f 1 -d ";")
  highest_error_code=$(echo $highest_error_code_for_module | cut -f 2 -d ";")
  
  error_codes_for_module=$(echo $error_codes | tr " " "\n" | cut -f 2,3 -d ";" | grep "$module" | cut -f 2 -d ";")
  
  # we assume find_error_codes.sh returns the error codes sorted by their module and number
  next_error_code=$(echo "$error_codes_for_module" | tr " " "\n" | awk 'BEGIN{ line=1000 } { if ($1 != line) { print line; exit } ; line++}') 
  
  if [ -n "$next_error_code" ]
  then
    echo $module-$next_error_code
  else
    echo $module-$((highest_error_code+1))
  fi
done
