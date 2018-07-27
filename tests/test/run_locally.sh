#!/bin/bash

set -u

die() { echo "ERROR:" "$@" >&2; exit 1; }

# $1: path to python test file
# $2: exasol server address
# $3...: additional arguments passed to tests
function run_test() {
    # echo "execute $@"
    export EXAPLUS=/opt/EXAplus-6.0.10/exaplus
    cmd=$(echo python -tt "$1" \
                      --driver=$(pwd)/../../downloads/EXASOL_ODBC-6.0.8/lib/linux/x86_64/libexaodbc-uo2214lv2.so \
                      --server "$2" \
                      --jdbc-path $(pwd)/../../EXASOL_JDBC-6.0.8/exajdbc.jar \
                      "${@:3}" # for, e.g., --lang
         )
    echo "$cmd"
    $cmd
}

# $1: run generic tests for lang $1, e.g., "java"
# $2: exasol server address
# $3: language definition
function run_generic_tests() {
    echo "Run generic language test for $1"
    for test in generic/*.py; do
        run_test "$test" "$2" --script-languages "$3" --lang "$1"
    done
}


# $1: run specific and generic tests for $1, e.g., "java"
# $2: exasol server address
# $3: language definition
function run_tests_in_folder() {
    folder="$1"
    echo "--- START TEST ${folder} ---"
    for test in $folder/*.py; do
        run_test "$test" "$2" --script-languages "$3"
    done
    echo "--- END TEST ${folder} ---"
}

optarr=$(getopt -o 'h' --long 'help,server:,test-config:' -- "$@")

eval set -- "$optarr"

# ./run-locally.sh --server 192... --test-config /home/...mini/../testconfig
while true; do
    case "$1" in
        --server) server="$2"; shift 2;;
        --test-config) test_config="$2"; shift 2;;
        --) shift; break;;
        *) echo "Usage: $0"
		       echo "Options:"
		       echo "  [--server=<host:port>]                Address of Exasol database instance"
		       echo "  [--test-config=<path>]                Path to flavor test config file"
		       echo "  [-h|--help]                           Print this help."; exit 0;;
    esac
done

if [ -z "$server" ]; then die "--server is required"; fi

if [ ! -f "$test_config" ]; then
    echo "testconfig for flavor $FLAVOR does not exist here: $config_file"
    exit 1
fi

typeset -A config
config=( )

while read line
do
    if echo $line | grep -F = &>/dev/null
    then
        varname=$(echo "$line" | cut -d '=' -f 1)
	      if [[ $varname == "#*" ]]; then continue; fi
        config[$varname]=$(echo "$line" | cut -d '=' -f 2-)
    fi
done < $test_config

for x in "${!config[@]}"; do printf "[%s]=%s\n" "$x" "${config[$x]}" ; done

if [ ! -z "${config[generic_language_tests]-}" ]; then
    for folder in ${config[generic_language_tests]}; do
        echo "$folder"
        run_generic_tests "$folder" "$server" "${config[language_definition]}"
    done
fi

if [ ! -z "${config[test_folders]-}" ]; then
    for folder in ${config[test_folders]}; do
        run_tests_in_folder "$folder" "$server" "${config[language_definition]}"
    done
fi

# if [ ! -z "${1-}" ]; then
#     run_tests_in_folder "$1" 2>&1 | tee "run_locally-$1.out"
#     exit $?
# else
#     languages=(java lua python r)
#     for lang in ${languages[@]}; do
#         run_tests_in_folder "$lang" 2>&1 | tee "run_locally-$lang.out"
#     done
# fi
