#!/bin/bash

set -euo pipefail

source /etc/os-release

if [[ "$VERSION_CODENAME" != "jammy" ]]; then
  echo "Error: Script can only work correctly under Ubuntu 22.04 (jammy)."
  exit 1
fi

if [[ $# -ne 1 ]]; then
  me=$(basename "$0")
  echo "Usage: $me <out_env_file>"
  echo "Example: $me .env"
  echo "<out_env_file> will contain environment variables needed to run 'bazel build'."
  exit 1
fi

OUT_ENV_FILE="$1"

export DEBIAN_FRONTEND=noninteractive

# Detect architecture
ARCH=$(dpkg --print-architecture)
case "$ARCH" in
  amd64)
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64"
    ;;
  arm64)
    BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64"
    ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

# Install dependencies
apt update
apt install -y \
  curl openjdk-11-jdk libzmq3-dev python3 protobuf-compiler \
  build-essential python3-pip libpcre3-dev libleveldb-dev chrpath tar \
  locales coreutils libssl-dev

# Install Bazelisk
install -d /usr/local/bin
curl -L "$BAZELISK_URL" -o /usr/local/bin/bazel
chmod +x /usr/local/bin/bazel

# Verify Bazelisk is working
bazel version

apt clean
apt autoremove -y

# Build and install SWIG 2.0.4
curl -L -o swig-2.0.4.tar.gz https://exasol-script-languages-dependencies.s3.eu-central-1.amazonaws.com/swig-2.0.4.tar.gz
tar zxf swig-2.0.4.tar.gz
(
  cd swig-2.0.4
  ./configure --prefix=/usr --build=$(arch)-unknown-linux-gnu 
  make -j"$(nproc)"
  make install
)
rm -rf swig-2.0.4 swig-2.0.4.tar.gz

# Install Python numpy package
pip install numpy

# Write environment variables to output file
cat > "$OUT_ENV_FILE" << EOF
export PYTHON3_PREFIX=/usr
export PYTHON3_VERSION=python3.10
export ZMQ_LIBRARY_PREFIX=/usr/lib/${ARCH}-linux-gnu/
export ZMQ_INCLUDE_PREFIX=/usr/include
export PROTOBUF_LIBRARY_PREFIX=/usr/lib/${ARCH}-linux-gnu
export PROTOBUF_INCLUDE_PREFIX=/usr/include/
export PROTOBUF_BIN=/usr/bin/protoc
export OPENSSL_LIBRARY_PREFIX=/usr/lib/${ARCH}-linux-gnu
export OPENSSL_INCLUDE_PREFIX=/usr/include/openssl
EOF
