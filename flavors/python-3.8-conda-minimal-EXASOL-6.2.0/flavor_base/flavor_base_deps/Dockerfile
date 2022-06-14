FROM {{ language_deps }}

RUN mkdir -p /build_info/packages
COPY flavor_base_deps/packages /build_info/packages/flavor_base_deps

RUN /scripts/install_scripts/install_via_conda.pl --file /build_info/packages/flavor_base_deps/conda_packages --channel-file /build_info/packages/flavor_base_deps/conda_channels --with-versions --conda-binary /opt/conda/bin/mamba
