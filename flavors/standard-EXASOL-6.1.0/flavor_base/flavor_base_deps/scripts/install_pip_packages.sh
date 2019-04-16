#!/bin/bash

pip_bin="$1"
input="$2"
while IFS= read -r var
do
    $pip_bin install "$var"
done < "$input"