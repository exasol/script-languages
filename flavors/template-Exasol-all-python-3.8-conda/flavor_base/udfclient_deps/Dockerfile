FROM {{ conda_deps }}

COPY udfclient_deps/packages /build_info/packages/udfclient_deps

RUN /scripts/install_scripts/install_via_conda.pl --file /build_info/packages/udfclient_deps/conda_packages --channel-file /build_info/packages/udfclient_deps/conda_channels --with-versions --conda-binary /bin/micromamba

ENV PROTOBUF_LIBRARY_PREFIX=$MAMBA_ROOT_PREFIX/lib
ENV PROTOBUF_INCLUDE_PREFIX=$MAMBA_ROOT_PREFIX/include
ENV ZMQ_LIBRARY_PREFIX=$MAMBA_ROOT_PREFIX/lib
ENV ZMQ_INCLUDE_PREFIX=$MAMBA_ROOT_PREFIX/include
