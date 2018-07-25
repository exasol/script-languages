#!/bin/bash
set -ux

# $1: path to python test file
# $2...: additional arguments
function run_test() {
    EXAPLUS=/opt/EXAplus-6.0.10/exaplus \
    python -tt "$1" \
           --driver /home/hece/nosync/EXASOL_ODBC-6.0.8/lib/linux/x86_64/libexaodbc-uo2214lv2.so \
           --server 192.168.122.241:8563 \
           --jdbc-path /home/hece/nosync/EXASOL_JDBC-5.0.8/exajdbc.jar \
           "${@:2}"
}
export -f run_test

# $1: run generic tests for lang $1, e.g., "java"
function run_generic_tests() {
    for test in generic/*.py; do
        run_test "$test" --lang $1
    done
}
function run_generic_tests_parallel() {
    find generic -iname '*py' | parallel "run_test {} --lang $1"
}
export -f run_generic_tests_parallel


# $1: run specific and generic tests for $1, e.g., "java"
run_tests_for_lang() {
    lang="$1"
    echo "--- START TEST ${lang} ---"
    for test in $lang/*.py; do
        run_test "$test"
    done
    run_generic_tests "$lang"
    echo "--- END TEST ${lang} ---"
}
run_tests_for_lang_parallel() {
    lang="$1"
    echo "--- START PARALLEL TEST ${lang} ---"
    find $lang -iname '*py' | parallel 'run_test {}'
    run_generic_tests_parallel "$lang"
    echo "--- END PARALLEL TEST ${lang} ---"
}


if [ ! -z "${1-}" ]; then
    run_tests_for_lang "$1" 2>&1 | tee "run_locally-$1.out"
    exit $?
fi

# ext-python
languages=(java lua mixed python r)
for lang in ${languages[@]}; do
    run_tests_for_lang "$lang" 2>&1 | tee "run_locally-$lang.out"
done
