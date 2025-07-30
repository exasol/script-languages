FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=""
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc
COPY scripts /scripts

RUN mkdir -p /build_info/packages
COPY build_deps/packages /build_info/packages/build_deps

# Install base dependencies
RUN /scripts/install_scripts/install_via_apt.pl \
    --file /build_info/packages/build_deps/apt_get_packages --with-versions

# Install Bazelisk (architecture-dependent)
ENV BAZELISK_VERSION="v1.26.0"
ENV USE_BAZEL_VERSION="7.6.1"

RUN set -e; \
    apt-get update && apt-get install -y curl gnupg; \
    ARCH=$(dpkg --print-architecture); \
    if [ "$ARCH" = "amd64" ]; then \
        BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64"; \
    elif [ "$ARCH" = "arm64" ]; then \
        BAZELISK_URL="https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-arm64"; \
    else \
        echo "Unsupported architecture: $ARCH" >&2; exit 1; \
    fi; \
    curl -fsSL -o /usr/local/bin/bazel "$BAZELISK_URL"; \
    chmod +x /usr/local/bin/bazel;
    
# Determine architecture and write relevant env vars to /env
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        JAVA_PREFIX="/usr/lib/jvm/java-11-openjdk-amd64"; \
        OPENSSL_LIB="/usr/lib/x86_64-linux-gnu"; \
    elif [ "$ARCH" = "arm64" ]; then \
        JAVA_PREFIX="/usr/lib/jvm/java-11-openjdk-arm64"; \
        OPENSSL_LIB="/usr/lib/aarch64-linux-gnu"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    echo "export JAVA_PREFIX=$JAVA_PREFIX" >> /env && \
    echo "export ARCH=$ARCH" >> /env && \
    echo "export PROTOBUF_LIBRARY_PREFIX=/usr/lib/" >> /env && \
    echo "export PROTOBUF_INCLUDE_PREFIX=/usr/include/" >> /env && \
    echo "export ZMQ_LIBRARY_PREFIX=/usr/lib" >> /env && \
    echo "export ZMQ_INCLUDE_PREFIX=/usr/include" >> /env && \
    echo "export OPENSSL_LIBRARY_PREFIX=$OPENSSL_LIB" >> /env && \
    echo "export OPENSSL_INCLUDE_PREFIX=/usr/include/openssl" >> /env && \
    echo "export PROTOBUF_BIN=/usr/bin/protoc" >> /env && \
    echo "export USE_BAZEL_VERSION=$USE_BAZEL_VERSION" >> /env

# Install SWIG 2.0.4
RUN curl -fsSL -o swig-2.0.4.tar.gz \
        https://exasol-script-languages-dependencies.s3.eu-central-1.amazonaws.com/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    (cd swig-2.0.4 && ./configure --build=$(arch)-unknown-linux-gnu --prefix=/usr && make -j"$(nproc)" && make install) && \
    rm -rf swig-2.0.4 swig-2.0.4.tar.gz

# Configure UTF-8 locale
RUN locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 && ldconfig
