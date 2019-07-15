#!/usr/bin/env bash

function parse_and_print () {
    package_version_list=$1
    for package_version in $package_version_list
    do
        package=$(echo "$package_version" | cut -f 1 -d ",")
        version=$(echo "$package_version" | cut -f 2 -d ",")
        echo "$package,$version"
    done
}

package_version_list_with_star=$(cat r-package-list.csv | grep '\*' | cut -f 1,3 -d " " | sed "s/ /,/")
parse_and_print "$package_version_list_with_star"

package_version_list_without_star=$(cat r-package-list.csv | grep -v '\*' | cut -f 1,2 -d " " | sed "s/ /,/")
parse_and_print "$package_version_list_without_star"