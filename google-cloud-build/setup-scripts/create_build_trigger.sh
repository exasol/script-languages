#!/usr/bin/env bash

PROJECT_ID=$(gcloud config get-value core/project --quiet)
JSON=$1
curl -s -XPOST -T ${JSON} -H"Authorization: Bearer $(gcloud config config-helper --format='value(credential.access_token)')" https://cloudbuild.googleapis.com/v1/projects/${PROJECT_ID}/triggers | jq '.'
