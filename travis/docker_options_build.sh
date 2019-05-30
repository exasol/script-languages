#!/usr/bin/env bash
TAG_PREFIX="${TRAVIS_REPO_SLUG//\//_}_$TRAVIS_BUILD_NUMBER"
SOURCE_OPTIONS="--source-docker-repository-name $BUILD_DOCKER_REPOSITORY --source-docker-username $BUILD_DOCKER_USERNAME --source-docker-password $BUILD_DOCKER_PASSWORD --source-docker-tag-prefix '$TAG_PREFIX'"
TARGET_OPTIONS="--target-docker-repository-name $BUILD_DOCKER_REPOSITORY --target-docker-username $BUILD_DOCKER_USERNAME --target-docker-password $BUILD_DOCKER_PASSWORD --target-docker-tag-prefix '$TAG_PREFIX'"
echo "$SOURCE_OPTIONS $TARGET_OPTIONS"