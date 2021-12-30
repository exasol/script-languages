#!/bin/bash

set -o pipefail
set -o errexit
set -o nounset

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

#shellcheck source=ext/scripts/install_scripts/parse_single_line_package_list_with_comments.sh
source "$SCRIPT_DIR/parse_single_line_package_list_with_comments.sh"

# This script installs a list of packages given by a file with a given command template

# Package Type, only for useful output necessary
package_type="$1"
# Command template, Example pip install <<package>>, the string <<package>> will be replaced by the actual package
command_template="$2"
# The file with the package list
package_list_file="$3"
if [[ -f "$package_list_file" ]]
then
    while IFS= read -r line || [ -n "$line" ]
    do
        package=$(parse_single_line_package_list_with_comments "$line")
        if [[ -n "$package" ]]
        then
            echo "$package_type: Installing package '$package'"
            command="${command_template/<<package>>/$package}"
            echo "Executing: $command"
            $command
        fi
    done < "$package_list_file"
    exit 0
else
    echo "Couldn't find $package_list_file"
    exit 1
fi