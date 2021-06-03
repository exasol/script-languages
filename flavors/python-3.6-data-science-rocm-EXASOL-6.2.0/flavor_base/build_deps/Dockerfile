FROM {{rocm_deps}}
ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=""
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc
RUN mkdir -p /build_info/packages
COPY build_deps/packages /build_info/packages/build_deps

ENV BAZEL_PACKAGE_VERSION="1.0.1"
ENV BAZEL_PACKAGE_FILE="bazel_$BAZEL_PACKAGE_VERSION-linux-x86_64.deb"
ENV BAZEL_PACKAGE_URL="https://github.com/bazelbuild/bazel/releases/download/$BAZEL_PACKAGE_VERSION/$BAZEL_PACKAGE_FILE"

RUN apt-get -y update && \
    apt-get install -y $(cat /build_info/packages/build_deps/apt_get_packages) && \
    curl -L --output "$BAZEL_PACKAGE_FILE" "$BAZEL_PACKAGE_URL" && \
    apt-get install -y "./$BAZEL_PACKAGE_FILE" && \
    rm "$BAZEL_PACKAGE_FILE" && \
    apt-get -y clean && \
    apt-get -y autoremove
	  
RUN curl -L -o swig-2.0.4.tar.gz https://storage.googleapis.com/exasol-script-languages-dependencies/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    (cd swig-2.0.4 && ./configure --prefix=/usr && make && make install) && \
    rm -rf swig-2.0.4 swig-2.0.4.tar.gz

RUN locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    ldconfig

RUN touch /env && \
    echo "export PROTOBUF_BIN=/usr/bin/protoc" >> /env && \
    echo "export JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64" >> /env
