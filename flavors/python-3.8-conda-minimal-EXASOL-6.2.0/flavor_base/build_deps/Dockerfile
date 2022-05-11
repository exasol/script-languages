FROM {{language_deps}}

RUN mkdir -p /build_info/packages
COPY build_deps/packages /build_info/packages/build_deps

RUN /scripts/install_scripts/install_via_conda.pl --file /build_info/packages/build_deps/conda_packages --with-versions

ENV PROTOBUF_BIN=/opt/conda/bin/protoc
