#!/usr/bin/env bash

function parse_single_line_package_list_with_comments(){
    # regex:
    #   begin with 0 or more whitespaces -> empty space in front
    #   than a group of 1 or more non whitespace characters -> package specification
    #   optional: end with a group of 1 or more whitespace followed by a # and than a sequence of any character
    #       -> comment in the end
    line="$1"
    package=$(echo "$line" | sed -r 's/^[ \t]*([^ \ŧ]+)([ \t]+#.*)?$/\1/')
    if [[ $package ]]
    then
        echo "$package"
    else
        echo "Ignored line $line" >&2
        echo
    fi
}