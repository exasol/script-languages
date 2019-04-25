#!/bin/bash

# This script installs a list of packages given by a file with a given command template

# Package Type, only for useful output necessary
package_type="$1"
# Command template, Example pip install <<list>>,
# the string <<list>> will be replaced by the packages wrapped by the package template
command_template="$2"
# Package template, wraps the package name, the string <<package>> gets replaced by the actual package name
package_template="$3"
# Separator between wrapped packages in the list
package_separator="$4"
# The file with the package list
package_list_file="$5"
if [[ -f "$package_list_file" ]]
then
    list=""
    while IFS= read -r package || [ -n "$package" ]
    do
        package=$(echo "$package" | cut -f 1 -d "#")
        package=$(echo "$package" | awk '{$1=$1;print}')
        if [[ -n "$package" ]]
        then
            echo "$package_type: Adding package '$package' to list"
            wrapped_package=$(echo "$package_template" | sed "s/<<package>>/$package/")
            if [[ -z "$list" ]]
            then
                list="$wrapped_package"
            else
                list="$list$package_separator$wrapped_package"
            fi
        fi
    done < "$package_list_file"
    echo "$package_type: Installing packages"
    if [[ -n "$list" ]]
    then
        command=$(echo "$command_template" | sed "s/<<list>>/$list/")
        echo "Executing: $command"
        if bash -c "$command"
        then
            echo "$package_type: Successfully installed packages"
        else
            echo "$package_type: Failed to install packages"
            exit 1
        fi
    else
        echo "Package list was empty"
    fi
    exit 0
else
    echo "Couldn't find $package_list_file"
    exit 1
fi