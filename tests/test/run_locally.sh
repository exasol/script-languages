#!/bin/bash
set -x
function run_test() {
    EXAPLUS=/opt/EXAplus-6.0.10/exaplus \
    python -tt "$1" \
           --driver /home/hece/nosync/EXASOL_ODBC-6.0.8/lib/linux/x86_64/libexaodbc-uo2214lv2.so \
           --server 192.168.122.12:8563 \
           --jdbc-path /home/hece/nosync/EXASOL_JDBC-5.0.8/exajdbc.jar \
           "${@:2}"
}

function run_generic_test() {
    for test in generic/*.py; do
        run_test "$test" --lang $1
    done
}

if [ ! -z "$1" ]; then
    run "$1"
    exit $?
fi

# ext-python
languages=(java) # java lua mixed python r)
for lang in ${languages[@]}; do
    echo "--- START TEST ${lang} ---"
    for test in $lang/*.py; do
        run_test "$test"
        run_generic_test "$lang"
    done
    echo "--- END TEST ${lang} ---"
done
