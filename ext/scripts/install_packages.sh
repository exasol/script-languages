#!/bin/bash

package_type="$1"
command="$2"
input="$3"
if [[ -f "$input" ]]
then
    while IFS= read -r package || [ -n "$package" ]
    do
        package=$(echo "$package" | cut -f 1 -d "#")
        package=$(echo "$package" | awk '{$1=$1;print}')
        if [[ -n "$package" ]]
        then
            echo "$package_type: Installing package '$package'"
            echo "Executing: $command '$package'"
            if $command "$package"
            then
                echo "$package_type: Successfully installed package '$package'"
            else
                echo "$package_type: Failed to install package '$package'"
                exit 1
            fi
        fi
    done < "$input"
    exit 0
else
    echo "Couldn't find $input"
    exit 1
fi