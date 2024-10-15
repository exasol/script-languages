#!/bin/bash

set -euo pipefail

source /etc/os-release

if [ "$VERSION_CODENAME" != "jammy" ]; then
  echo "Script can only work correctly under Ubuntu 22.04"
  exit 1
fi

if [ $# -ne 1 ];
then
  me=$(basename "$0")
  echo "Usage: '$me <out_env_file>'. For example: '$me .env'. <out_env_file> will contain necessary environment variables to run 'bazel build'."
  exit 1
fi

OUT_ENV_FILE="$1"

export DEBIAN_FRONTEND=noninteractive

apt update
apt install -y curl openjdk-11-jdk libzmq3-dev python3 protobuf-compiler build-essential python3-pip libpcre3-dev chrpath tar locales coreutils libssl-dev

BAZEL_PACKAGE_VERSION="7.2.1"
BAZEL_PACKAGE_FILE="bazel_$BAZEL_PACKAGE_VERSION-linux-x86_64.deb"
BAZEL_PACKAGE_URL="https://github.com/bazelbuild/bazel/releases/download/$BAZEL_PACKAGE_VERSION/$BAZEL_PACKAGE_FILE"


curl -L --output "$BAZEL_PACKAGE_FILE" "$BAZEL_PACKAGE_URL" && \
apt install -y "./$BAZEL_PACKAGE_FILE" && \
rm "$BAZEL_PACKAGE_FILE" && \

apt -y clean
apt -y autoremove


curl -L -o swig-2.0.4.tar.gz https://exasol-script-languages-dependencies.s3.eu-central-1.amazonaws.com/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    (cd swig-2.0.4 && ./configure --prefix=/usr && make && make install) && \
    rm -rf swig-2.0.4 swig-2.0.4.tar.gz


pip install numpy


cat >"$OUT_ENV_FILE" <<EOL
export PYTHON3_PREFIX=/usr
export PYTHON3_VERSION=python3.10
export ZMQ_LIBRARY_PREFIX=/usr/lib/x86_64-linux-gnu/
export ZMQ_INCLUDE_PREFIX=/usr/include
export PROTOBUF_LIBRARY_PREFIX=/usr/lib/x86_64-linux-gnu
export PROTOBUF_INCLUDE_PREFIX=/usr/include/
export PROTOBUF_BIN=/usr/bin/protoc
EOL
