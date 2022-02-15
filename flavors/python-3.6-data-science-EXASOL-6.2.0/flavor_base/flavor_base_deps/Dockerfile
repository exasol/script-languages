FROM {{ language_deps }}

RUN mkdir -p /build_info/packages
COPY flavor_base_deps/packages /build_info/packages/flavor_base_deps

RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/flavor_base_deps/apt_get_packages --with-versions

# Need to run with --allow-no-version, as many packages datascience package are not pinned
RUN /scripts/install_scripts/install_via_pip.pl --file /build_info/packages/flavor_base_deps/python3_pip_packages --python-binary python3 --with-versions --allow-no-version  --use-deprecated-legacy-resolver --ignore-installed
