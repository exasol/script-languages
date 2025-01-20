#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

oft_version="4.1.0"

base_dir="$1"
shift 1
src_dir="$1"
shift 1
additional_options=$@
readonly base_dir
readonly oft_jar="$HOME/.m2/repository/org/itsallcode/openfasttrace/openfasttrace/$oft_version/openfasttrace-$oft_version.jar"

if [ ! -f "$oft_jar" ]; then
    echo "Downloading OpenFastTrace $oft_version"
    mvn --batch-mode org.apache.maven.plugins:maven-dependency-plugin:3.3.0:get -Dartifact=org.itsallcode.openfasttrace:openfasttrace:$oft_version
fi

# Trace all
java -jar "$oft_jar" trace \
    $additional_options \
    -a feat,req,dsn \
    "$base_dir/docs" \
    "$base_dir/$src_dir"
