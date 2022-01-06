#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

# Usage
# show_code_arround_error_codes.sh <regex-pattern1> <regex-pattern2>
# pattern filter line of <filename>:<linenumber>|<error-code>
# multiple pattern get merged with | to an or construction

if [ $# -gt 0 ]
then
  args="($(echo "$@" | tr " " "|"))"
else
  args=".*"
fi
echo "$args"
error_codes=$(bash find_error_codes.sh | awk '{print $1 "|" $2 "-" $3 "-" $4}' | grep -E "$args")

echo "$error_codes" \
  | tr " " "\n" \
  | tr "|" " " \
  | tr ":" " " \
  | awk 'BEGIN{window=10;line_sep=sprintf("%80s","");gsub(/ /,"=",line_sep)}{min=$2-window; if (min<1){min=1}; print "echo " line_sep " ; echo == " $3 " == " $1 ":" $2 " ; echo " line_sep " ; cat " $1 " | nl -ba -w8 | sed \\\"" $2 "s/^  /->/\\\" | sed -n " min "," $2+window "p ; echo"}' \
  | xargs -n 1 -I {} bash -c '{}'
#  | xargs -n 1 -I {} echo "bash -c '{}'"
