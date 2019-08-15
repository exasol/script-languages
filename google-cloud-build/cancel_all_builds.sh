#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
gcloud builds list | grep WORKING | cut -f 1 -d " " | xargs gcloud builds cancel
