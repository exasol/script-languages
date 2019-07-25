#!/usr/bin/env bash
TAG_PREFIX="${TRAVIS_REPO_SLUG//\//_}_$TRAVIS_BUILD_NUMBER"
SOURCE_OPTIONS="--source-docker-repository-name '$BUILD_DOCKER_REPOSITORY' --source-docker-username '$BUILD_DOCKER_USERNAME' --source-docker-password '$BUILD_DOCKER_PASSWORD'  --source-docker-tag-prefix '$TAG_PREFIX'"
TARGET_OPTIONS="--target-docker-repository-name '$DEPLOY_DOCKER_REPOSITORY' --target-docker-username '$DEPLOY_DOCKER_USERNAME' --target-docker-password '$DEPLOY_DOCKER_PASSWORD'"
echo "$SOURCE_OPTIONS $TARGET_OPTIONS"