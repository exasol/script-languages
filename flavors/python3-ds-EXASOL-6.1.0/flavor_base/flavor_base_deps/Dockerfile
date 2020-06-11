FROM {{ language_deps }}

RUN mkdir -p /build_info/packages
COPY flavor_base_deps/packages /build_info/packages/flavor_base_deps

RUN /scripts/install_scripts/install_scripts/install_via_apt.pl --file /build_info/packages/flavor_base_deps/apt_get_packages

RUN /scripts/install_scripts/install_scripts/install_via_pip.pl --file /build_info/packages/flavor_base_deps/pip3_packages --python-binary python3
