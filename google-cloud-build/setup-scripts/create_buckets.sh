#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

function check_output(){
  if [[ "$OUTPUT" =~ ".*Exception.*" ]]
  then
    if ! [[ "$OUTPUT" =~ ".*already exists.*" ]]
    then
      echo "$OUTPUT"
      exit 1
    else
      echo "Already exists"
    fi
  else
    echo "Done"
  fi 
}

env_file=".env/env.yaml"
LOG_BUCKET=$(cat "$env_file" | yq -r .log_bucket)
CONTAINER_BUCKET=$(cat "$env_file" | yq -r .container_bucket)

echo "Create Log Bucket $LOG_BUCKET"
OUTPUT=$(gsutil mb $LOG_BUCKET 2>&1 || true)
check_output
echo "Create Container Bucket $CONTAINER_BUCKET"
OUTPUT=$(gsutil mb $CONTAINER_BUCKET 2>&1 || true)
check_output
