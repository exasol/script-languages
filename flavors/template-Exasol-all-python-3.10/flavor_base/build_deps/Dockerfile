FROM exasol/slc_base:22.04
ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=""
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc
COPY scripts /scripts

RUN mkdir -p /build_info/packages


COPY build_deps_packages.yml /build_info/packages/build_deps_packages.yml
RUN exaslpm install --package-file /build_info/packages/build_deps_packages.yml --build-step build_deps

RUN curl -L -o swig-2.0.4.tar.gz https://exasol-script-languages-dependencies.s3.eu-central-1.amazonaws.com/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    (cd swig-2.0.4 && ./configure --prefix=/usr && make && make install) && \
    rm -rf swig-2.0.4 swig-2.0.4.tar.gz

RUN locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    ldconfig

RUN touch /env && \
    echo "export PROTOBUF_BIN=/usr/bin/protoc" >> /env && \
    echo "export JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64" >> /env
