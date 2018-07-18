#!/bin/bash
set -x
function run() {
    EXAPLUS=/opt/EXAplus-6.0.10/exaplus \
    python -tt "$1" \
           --driver /home/hece/nosync/EXASOL_ODBC-6.0.8/lib/linux/x86_64/libexaodbc-uo2214lv2.so \
           --server 192.168.122.12:8563 \
           --jdbc-path /home/hece/nosync/EXASOL_JDBC-5.0.8/exajdbc.jar
}

if [ ! -z "$1" ]; then
    run "$1"
    exit $?
fi

# ext-python
languages=(generic) # java lua mixed python r)
for lang in ${languages[@]}; do
    echo "--- START TEST ${lang} ---"
    for test in $lang/*.py; do
        EXAPLUS=/opt/EXAplus-6.0.10/exaplus \
               python -tt "$test" \
                --lang=python \
                --driver /home/hece/nosync/EXASOL_ODBC-6.0.8/lib/linux/x86_64/libexaodbc-uo2214lv2.so \
                --server 192.168.122.12:8563 \
                --jdbc-path /home/hece/nosync/EXASOL_JDBC-5.0.8/exajdbc.jar
    done
    echo "--- END TEST ${lang} ---"
done
