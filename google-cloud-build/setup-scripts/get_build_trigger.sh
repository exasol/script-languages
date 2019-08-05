#!/usr/bin/env bash

PROJECT_ID=$(gcloud config get-value core/project --quiet)
TRIGGER_ID=$1
curl -s -XGET -H"Authorization: Bearer $(gcloud config config-helper --format='value(credential.access_token)')" https://cloudbuild.googleapis.com/v1/projects/${PROJECT_ID}/triggers/${TRIGGER_ID} | jq '.'

