#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
LOG_BUCKET="$1"
FLAVOR="$2"
BUILD_ID="$3"
DATETIME_FILE="datetime.txt"
if [ ! -f "$DATETIME_FILE" ]
then
	date --utc +%Y%m%d_%H%M%S > "$DATETIME_FILE"
fi
DATETIME=$(cat $DATETIME_FILE)
BUCKET="$LOG_BUCKET/build_output/$FLAVOR/${DATETIME}_${BUILD_ID}/"
gsutil rsync -C -x exports -r .build_output "$BUCKET" &> rync.log || echo "fail" > /workspace/build-status.txt 
gsutil cp rync.log "$BUCKET"
if [[ $(< /workspace/build-status.txt) == "fail" ]]; then
	exit 1
fi
