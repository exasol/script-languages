FROM {{udfclient_deps}}

RUN mkdir -p /build_info/packages/language_deps
COPY language_deps/packages/apt_get_packages /build_info/packages/language_deps

RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/language_deps/apt_get_packages --with-versions

RUN /scripts/install_scripts/install_python3.6_pip.sh

COPY language_deps/packages/python3_pip_packages /build_info/packages/language_deps
# Install Python3 packages with --allow-no-version as pandas/numpy are not pinned.
RUN /scripts/install_scripts/install_via_pip.pl --file /build_info/packages/language_deps/python3_pip_packages --python-binary python3 -with-versions --allow-no-version

ENV PYTHON3_PREFIX /usr
ENV PYTHON3_VERSION python3.6

