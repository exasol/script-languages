#!/bin/bash
set -u
set -o errexit
set -o pipefail

function run_wrapper(){
    local START_DATE_TIME="$(date '+%Y_%m_%d_%H_%M_%S')"

    # We get called with the following commandline arguments 
    # <socket> lang=python|lang=r|lang=java|lang=streaming|lang=benchmark path/to/UDFClientExecutable path/to/wrapper_script.sh <WRAPPER_SCRIPT_SPECIFIC_OPTIONS>
    local SOCKET_PATH="${BASH_ARGV[1]}"
    local LANGUAGE="${BASH_ARGV[2]}"
    local UDFCLIENT_EXECUTABLE="${BASH_ARGV[3]}"
    local WRAPPER_SCRIPT_PATH="${BASH_ARGV[4]}"
    local WRAPPER_SCRIPT_DIR="$(dirname "$(readlink -f "$WRAPPER_SCRIPT_PATH")")"
    local CONNECTION_NAME="$(echo $NAME | sed s#[/:]##g)"
    local TEMPORARY_BASE_DIRECTORY="/tmp/wrapper_tmp_$START_DATE_TIME_$CONNECTION_NAME/" 
    mkdir -p "$TEMPORARY_BASE_DIRECTORY"

    source "$WRAPPER_SCRIPT_PATH"
    
    before_udfclient_execution("$CONNECTION_NAME","$START_DATE_TIME","$WRAPPER_SCRIPT_DIR")
    wrap_udfclient_execution("$UDFCLIENT_EXECUTABLE","$SOCKET_PATH $LANGUAGE")
    after_udfclient_execution()
}

local RUNNER_SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
run_wrapper("$RUNNER_SCRIPT_DIR")

