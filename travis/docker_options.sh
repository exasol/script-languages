#!/usr/bin/env bash
SOURCE_OPTIONS="--source-docker-repository-name $DOCKER_REPOSITORY --source-docker-username $DOCKER_USERNAME --source-docker-password $DOCKER_PASSWORD --source-docker-tag-prefix '$TRAVIS_BUILD_NUMBER'"
TARGET_OPTIONS="--target-docker-repository-name $DOCKER_REPOSITORY --target-docker-username $DOCKER_USERNAME --target-docker-password $DOCKER_PASSWORD --target-docker-tag-prefix '$TRAVIS_BUILD_NUMBER'"
echo "$SOURCE_OPTIONS $TARGET_OPTIONS"