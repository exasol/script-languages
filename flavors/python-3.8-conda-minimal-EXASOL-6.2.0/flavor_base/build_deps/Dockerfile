FROM {{language_deps}}

RUN mkdir -p /build_info/packages
COPY build_deps/packages /build_info/packages/build_deps

RUN /scripts/install_scripts/install_via_conda.pl --file /build_info/packages/build_deps/conda_packages --channel-file /build_info/packages/build_deps/conda_channels --with-versions

RUN curl -L -o swig-2.0.4.tar.gz https://storage.googleapis.com/exasol-script-languages-dependencies/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    (cd swig-2.0.4 && ./configure --prefix=/usr && make && make install) && \
    rm -rf swig-2.0.4 swig-2.0.4.tar.gz

ENV PROTOBUF_BIN=/opt/conda/bin/protoc
