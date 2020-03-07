#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail
DOCKER_USER=$2
DOCKER_PASSWORD="$(cat secrets/DOCKER_PASSWORD)"
echo "$DOCKER_PASSWORD" | docker login --username "$DOCKER_USER" --password-stdin

#set -x

DOCKER_REPOSITORY=$1
DOCKERFILE_HASH=$(cat google-cloud-build/ci-scripts/scripts/Dockerfile | shasum | cut -f 1 -d " ")
TAG_NAME="build-container-$DOCKERFILE_HASH"
TAG_NAME_SHORT="${TAG_NAME:0:127}"
IMAGE_NAME="$DOCKER_REPOSITORY:$TAG_NAME_SHORT"

echo
echo "=========================================================="
echo "Try to pull image $IMAGE_NAME"
echo "=========================================================="
echo
if ! docker pull "$IMAGE_NAME"
then
  echo
  echo "=========================================================="
  echo "Pull failed going to build image $IMAGE_NAME"
  echo "=========================================================="
  echo
  docker build -t "$IMAGE_NAME" google-cloud-build/ci-scripts/scripts/

  echo
  echo "=========================================================="
  echo "Push image $IMAGE_NAME"
  echo "=========================================================="
  echo
  docker push "$IMAGE_NAME"
fi

LOCAL_IMAGE_NAME="exasol/script-languages:build-container"
echo
echo "=========================================================="
echo "Rename image $IMAGE_NAME" to "$LOCAL_IMAGE_NAME"
echo "=========================================================="
echo
docker tag "$IMAGE_NAME" "$LOCAL_IMAGE_NAME"

echo
echo "=========================================================="
echo "Printing docker images"
echo "=========================================================="
echo
docker images | grep exa

