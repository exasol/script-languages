#!/usr/bin/env bash
SOURCE_TAG_PREFIX="${TRAVIS_REPO_SLUG//\//_}_$TRAVIS_BUILD_NUMBER"
TARGET_TAG_PREFIX="${TRAVIS_REPO_SLUG//\//_}_$TRAVIS_BRANCH"
SOURCE_OPTIONS="--source-docker-repository-name $BUILD_DOCKER_REPOSITORY --source-docker-username $BUILD_DOCKER_USERNAME --source-docker-password $BUILD_DOCKER_PASSWORD --source-docker-tag-prefix '$SOURCE_TAG_PREFIX'"
TARGET_OPTIONS="--target-docker-repository-name $BUILD_DOCKER_REPOSITORY --target-docker-username $BUILD_DOCKER_USERNAME --target-docker-password $BUILD_DOCKER_PASSWORD --target-docker-tag-prefix '$TARGET_TAG_PREFIX'"
echo "$SOURCE_OPTIONS $TARGET_OPTIONS"