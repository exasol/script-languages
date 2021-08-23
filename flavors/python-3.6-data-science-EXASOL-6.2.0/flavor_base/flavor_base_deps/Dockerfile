FROM {{ language_deps }}

RUN mkdir -p /build_info/packages/flavor_base_deps

COPY flavor_base_deps/packages/apt_get_packages /build_info/packages/flavor_base_deps
RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/language_deps/apt_get_packages --with-versions --allow-no-version

COPY flavor_base_deps/packages/python3_pip_packages /build_info/packages/flavor_base_deps
RUN /scripts/install_scripts/install_via_pip.pl --file /build_info/packages/flavor_base_deps/python3_pip_packages --python-binary python3 --with-versions --allow-no-version
