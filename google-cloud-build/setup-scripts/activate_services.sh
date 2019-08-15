#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
gcloud services enable sourcerepo.googleapis.com
gcloud services enable cloudapis.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable servicemanagement.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable pubsub.googleapis.com
gcloud services enable cloudkms.googleapis.com
