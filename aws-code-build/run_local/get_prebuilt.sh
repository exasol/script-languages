#!/bin/bash
set -euo pipefail

pushd "../.." &>/dev/null

echo Pulling codebuild script...
wget https://raw.githubusercontent.com/aws/aws-codebuild-docker-images/master/local_builds/codebuild_build.sh
docker pull exadockerci4/aws_codebuild:5.0
echo Ready.
echo Now you can run the AWS code build locally with:
echo "'./aws-code-build/run_local/run_local_prebuilt.sh <buildspec> <aws-profile> in the root folder"

popd &> /dev/null
