#!/bin/bash
set -o nounset
set -o errexit
set -o pipefail

function check_output(){
  if echo "$OUTPUT" | grep -q "ERROR" && \
    ! echo "$OUTPUT" | grep -q "already exists"
  then
    echo "===================================="
    echo "$OUTPUT"
    echo "===================================="
    exit -1
  fi
}

env_file=".env/env.yaml"
KEY_RING_NAME=$(yq -r .key_ring_name < "$env_file")
KEY_NAME=$(yq -r .key_name < "$env_file")
PROJECT_NAME=$(yq -r .gcloud_project_name < "$env_file")

echo "Create key ring"
OUTPUT=$(gcloud kms keyrings create "$KEY_RING_NAME" --location global 2>&1 || true)
check_output

echo "Create key"
OUTPUT=$(gcloud kms keys create "$KEY_NAME" --location global --keyring "$KEY_RING_NAME" --purpose encryption 2>&1 | grep "already exists" || true)
check_output

echo "Get project id"
PROJECT=$(yq -r .gcloud_project_name < "$env_file")
echo "PROJECT_NAME: $PROJECT"
PROJECT_NUMBER=$(gcloud projects list --filter="$PROJECT" --format="value(PROJECT_NUMBER)")
echo "PROJECT_NUMBER: $PROJECT_NUMBER"


echo "Grant key to Cloud Build"
OUTPUT=$(gcloud kms keys add-iam-policy-binding "$KEY_NAME" --location=global --keyring="$KEY_RING_NAME" --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" --role=roles/cloudkms.cryptoKeyDecrypter 2>&1 || true)
check_output

echo "Create service account for Build Cloud Functions"
OUTPUT=$(gcloud beta iam service-accounts create build-cloud-functions --display-name "build-cloud-functions" 2>&1 || true)
check_output

echo "Grant key to Cloud Functions"
OUTPUT=$(gcloud kms keys add-iam-policy-binding "$KEY_NAME" --location=global --keyring="$KEY_RING_NAME" --member="serviceAccount:build-cloud-functions@$PROJECT_NAME.iam.gserviceaccount.com"  --role=roles/cloudkms.cryptoKeyDecrypter 2>&1 || true)
check_output
