#!/bin/bash

let rc=0
for t in $(dirname $0)/*/*.py; do
    if [ ! -e $t ]; then
        continue
    fi
    $t --lint
    let rc=rc+$?
    rm $(basename ${t%.py}).log
done
echo $rc errors;
test $rc -eq 0;

# vim: ts=4:sts=4:sw=4:et:fdm=indent
