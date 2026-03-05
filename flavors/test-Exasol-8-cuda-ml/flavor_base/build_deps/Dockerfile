FROM {{language_deps}}

RUN mkdir -p /build_info/packages

COPY build_deps_packages.yml /build_info/packages/build_deps_packages.yml
RUN exaslpm install --package-file /build_info/packages/build_deps_packages.yml --build-step build_deps

RUN curl -L -o swig-2.0.4.tar.gz https://exasol-script-languages-dependencies.s3.eu-central-1.amazonaws.com/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    (cd swig-2.0.4 && ./configure --prefix=/usr && make && make install) && \
    rm -rf swig-2.0.4 swig-2.0.4.tar.gz

ENV PROTOBUF_BIN=$MAMBA_ROOT_PREFIX/bin/protoc
