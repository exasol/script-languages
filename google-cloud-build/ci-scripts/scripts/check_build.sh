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
BUCKET="$LOG_BUCKET/$FLAVOR/${DATETIME}_${BUILD_ID}/"
BUILD_OUTPUT_PATH=".build_output/jobs"
echo "=========================================================="
echo "Copy $BUILD_OUTPUT_PATH to $BUCKET"
echo "=========================================================="
echo
gsutil -m rsync -C -x exports -r "$BUILD_OUTPUT_PATH" "$BUCKET" 2>&1 | tee rync.log || echo "fail" > /workspace/build-status.txt 
echo
echo "=========================================================="
echo "Copy rsync.log to $BUCKET/rsync.log"
echo "=========================================================="
echo
gsutil cp rync.log "$BUCKET"
if [[ $(< /workspace/build-status.txt) == "fail" ]]; then
	exit 1
fi
