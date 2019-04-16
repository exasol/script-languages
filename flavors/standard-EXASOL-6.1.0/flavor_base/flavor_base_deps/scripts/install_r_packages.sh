#!/bin/bash

input="$1"
while IFS= read -r var
do
    Rscript -e "$var"
done < "$input"