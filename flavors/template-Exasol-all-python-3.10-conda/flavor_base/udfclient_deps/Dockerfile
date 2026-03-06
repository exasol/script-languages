FROM {{ conda_deps }}


COPY udfclient_deps_packages.yaml /build_info/packages/udfclient_deps_packages.yaml
RUN exaslpm install --package-file /build_info/packages/udfclient_deps_packages.yaml --build-step udfclient_deps

ENV PROTOBUF_LIBRARY_PREFIX=$MAMBA_ROOT_PREFIX/lib
ENV PROTOBUF_INCLUDE_PREFIX=$MAMBA_ROOT_PREFIX/include
ENV ZMQ_LIBRARY_PREFIX=$MAMBA_ROOT_PREFIX/lib
ENV ZMQ_INCLUDE_PREFIX=$MAMBA_ROOT_PREFIX/include
