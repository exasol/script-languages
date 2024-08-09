#!/bin/bash

#Reason: .env file does not exist in repo. Need to disable shellcheck rule.
#shellcheck disable=SC1091
source .env
bash build.sh "$@"