#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
TYPE=$1
DOCKER_REPOSITORY=$2
DOCKER_USERNAME=$3
TAG_PREFIX=$4
DOCKER_OPTIONS=""
if [ -n "$DOCKER_REPOSITORY" ] && [ "$DOCKER_REPOSITORY" != '""' ]
then
  DOCKER_OPTIONS="$DOCKER_OPTIONS --$TYPE-docker-repository-name $DOCKER_REPOSITORY"
fi
if [ -n "$DOCKER_USERNAME" ] && [ "$DOCKER_USERNAME" != '""' ]
then
  DOCKER_OPTIONS="$DOCKER_OPTIONS --$TYPE-docker-username $DOCKER_USERNAME"
fi
if [ -n "$TAG_PREFIX" ] && [ "$TAG_PREFIX" != '""' ]
then
  DOCKER_OPTIONS="$DOCKER_OPTIONS --$TYPE-docker-tag-prefix $TAG_PREFIX"
fi
echo $DOCKER_OPTIONS
