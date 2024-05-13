FROM {{ language_deps }}

RUN mkdir -p /build_info/packages
COPY flavor_base_deps/packages /build_info/packages/flavor_base_deps

ENV CONDA_OVERRIDE_CUDA=11.3

RUN /scripts/install_scripts/install_via_conda.pl --file /build_info/packages/flavor_base_deps/conda_packages --channel-file /build_info/packages/flavor_base_deps/conda_channels --with-versions --conda-binary "$MAMBA_ROOT_PREFIX/bin/mamba"

RUN /scripts/install_scripts/install_via_pip.pl --file /build_info/packages/flavor_base_deps/python3_pip_packages --python-binary "$MAMBA_ROOT_PREFIX/bin/python3.8" --with-versions --allow-no-version
