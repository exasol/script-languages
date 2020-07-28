#!/bin/bash
set -u
set -o errexit
set -o pipefail


# Function which gets called before the UDFClient execution and should save any necessary information from its parameters in global variables for the subsequent functions
# Parameters:
#   CONNECTION_NAME             The connection name for this UDF instance safe for the use in filesystem paths (may not unique overtime, to be sure include startup time)
#   START_DATE_TIME             The start time of this UDF instance safe for the use in filesystem_paths (+%Y_%m_%d_%H_%M_%S)
#   TEMPORARY_BASE_DIRECTORY    The temporary base directory which you should use for all files you want to save until the end if the instance
#   WRAPPER_SCRIPT_DIR          The directory from where this wrapper scripts was loaded
#   WRAPPER_SPECIFIC_PARAMETERS Additional parameter for the wrapper script which were appended to the runner call
function before_udfclient_execution(){
    local CONNECTION_NAME="$1"
    local START_DATE_TIME="$2"
    local TEMPORARY_BASE_DIRECTORY="$3"
    local WRAPPER_SCRIPT_DIR="$4"
    source "$WRAPPER_SCRIPT_DIR/config.sh"
    MASSIF_FILE_NAME="MASSIF_${START_DATE_TIME}_${CONNECTION_NAME}"
    MASSIF_FILE_PATH="${TEMPORARY_BASE_DIRECTORY}/${MASSIF_FILE_NAME}"
}

# Function which gets the command to call the UDFClient, wrap it and than call it
# Parameters:
#   UDFCLIENT_EXECUTABLE
#   UDFCLIENT_PARAMETERS
function wrap_udfclient_execution(){
    local UDFCLIENT_EXECUTABLE="$1"
    local UDFCLIENT_PARAMETERS="$2"
    valgrind --tool=massif --massif-out-file='$MASSIF_FILE' "$UDFCLIENT_EXECUTABLE" $UDFCLIENT_PARAMETERS
}

# Function which gets called after the UDFClient execution (here it is time to permamently save your debug output, either through the BucketFS or via standard out and the SCRIPT_OUTPUT_REDIRECT)
function after_udfclient_execution(){
    curl -v --fail -X PUT -T "${MASSIF_FILE_PATH}" "${BUCKETFS_UPLOAD_URL_PREFIX}/${MASSIF_FILE_NAME}"
}

