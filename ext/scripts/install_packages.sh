#!/bin/bash

# This script installs a list of packages given by a file with a given command template

# Package Type, only for useful output necessary
package_type="$1"
# Command template, Example pip install <<package>>, the string <<package>> will be replaced by the actual package
command_template="$2"
# The file with the package list
package_list_file="$3"
if [[ -f "$package_list_file" ]]
then
    while IFS= read -r package || [ -n "$package" ]
    do
        package=$(echo "$package" | cut -f 1 -d "#")
        package=$(echo "$package" | awk '{$1=$1;print}')
        if [[ -n "$package" ]]
        then
            echo "$package_type: Installing package '$package'"
            command=$(echo "$command_template" | sed "s/<<package>>/$package/")
            echo "Executing: $command"
            if $command
            then
                echo "$package_type: Successfully installed package '$package'"
            else
                echo "$package_type: Failed to install package '$package'"
                exit 1
            fi
        fi
    done < "$package_list_file"
    exit 0
else
    echo "Couldn't find $package_list_file"
    exit 1
fi