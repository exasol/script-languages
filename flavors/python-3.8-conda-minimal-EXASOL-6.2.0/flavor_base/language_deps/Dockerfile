FROM {{udfclient_deps}}

RUN mkdir -p /build_info/packages/language_deps
COPY language_deps/packages/ /build_info/packages/language_deps

RUN /scripts/install_scripts/install_via_conda.pl --file /build_info/packages/language_deps/conda_packages --channel-file /build_info/packages/language_deps/conda_channels --with-versions

ENV PYTHON3_PREFIX /opt/conda
ENV PYTHON3_VERSION python3.8

