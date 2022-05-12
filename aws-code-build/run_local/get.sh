#!/bin/bash
set -euo pipefail

pushd "../.." &>/dev/null

echo Pulling codebuild script...
wget https://raw.githubusercontent.com/aws/aws-codebuild-docker-images/master/local_builds/codebuild_build.sh
if [[ ! $(docker image ls aws/codebuild/ubuntu/standard:5.0 --format="true") ]] ;
then
 echo "pulling https://github.com/aws/aws-codebuild-docker-images"
 pushd /tmp
 git clone https://github.com/aws/aws-codebuild-docker-images
 pushd aws-codebuild-docker-images/ubuntu/standard/5.0
 echo "Building aws/codebuild/ubuntu/standard:5.0"
 docker build -t aws/codebuild/ubuntu/standard:5.0 .
 popd
 rm -rf aws-codebuild-docker-images
 popd
fi

echo Ready.
echo Now you can run the AWS code build locally with:
echo "'./aws-code-build/run_local/run_local.sh in the root folder (where buildspec.yml exists)"

popd &> /dev/null
