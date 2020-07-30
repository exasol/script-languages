#!/bin/bash
set -u
set -o errexit
set -o pipefail

function run_wrapper(){
    local START_DATE_TIME
    START_DATE_TIME="$(date '+%Y_%m_%d_%H_%M_%S')"

    # We get called with the following commandline arguments 
    # <socket> lang=python|lang=r|lang=java|lang=streaming|lang=benchmark path/to/UDFClientExecutable path/to/wrapper_script.sh <WRAPPER_SCRIPT_SPECIFIC_PARAMETERS>
    
    local SOCKET_PATH=
    SOCKET_PATH="$1"
    shift 1
    
    local LANGUAGE
    LANGUAGE="$1"
    shift 1
    
    local UDFCLIENT_EXECUTABLE
    UDFCLIENT_EXECUTABLE="$1"
    shift 1

    local WRAPPER_SCRIPT_PATH
    WRAPPER_SCRIPT_PATH="$1"
    shift 1

    local WRAPPER_SCRIPT_SPECIFIC_PARAMETERS
    WRAPPER_SCRIPT_SPECIFIC_PARAMETERS=( "$@" )

    local WRAPPER_SCRIPT_DIR
    WRAPPER_SCRIPT_DIR="$(dirname "$(readlink -f "$WRAPPER_SCRIPT_PATH")")"

    local CONNECTION_NAME
    CONNECTION_NAME="${SOCKET_PATH//[^a-zA-Z0-9]/_}"
    
    local TEMPORARY_BASE_DIRECTORY
    TEMPORARY_BASE_DIRECTORY="/tmp/wrapper_tmp_${START_DATE_TIME}_${CONNECTION_NAME}" 

    local UDFCLIENT_PARAMETERS
    UDFCLIENT_PARAMETERS=( "$SOCKET_PATH" "$LANGUAGE" )

    mkdir -p "$TEMPORARY_BASE_DIRECTORY"

    # shellcheck source=debug_udfclient_wrapper_stdouterr_to_bucketfs.sh
    source "$WRAPPER_SCRIPT_PATH"
    
    before_udfclient_execution "$CONNECTION_NAME" "$START_DATE_TIME" "$TEMPORARY_BASE_DIRECTORY" "$WRAPPER_SCRIPT_DIR" "${WRAPPER_SCRIPT_SPECIFIC_PARAMETERS[@]}"
    wrap_udfclient_execution "$UDFCLIENT_EXECUTABLE" "${UDFCLIENT_PARAMETERS[@]}"
    after_udfclient_execution
}

run_wrapper "$@"
