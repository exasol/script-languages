FROM {{udfclient_deps}}

RUN mkdir -p /build_info/packages/language_deps
COPY language_deps/packages/apt_get_packages /build_info/packages/language_deps
RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/language_deps/apt_get_packages --with-versions

RUN /scripts/install_scripts/install_key.pl --key-server hkp://keyserver.ubuntu.com:80 --key F23C5A6CF475977595C89F51BA6932366A755776 && \
    /scripts/install_scripts/install_ppa.pl --ppa 'deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic main' --out-file bionic-deadsnakes

COPY language_deps/packages/apt_get_packages_python /build_info/packages/language_deps
RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/language_deps/apt_get_packages_python --with-versions

RUN /scripts/install_scripts/install_python3.7_pip.sh

COPY language_deps/packages/python3_pip_packages /build_info/packages/language_deps
RUN /scripts/install_scripts/install_via_pip.pl --file /build_info/packages/language_deps/python3_pip_packages --python-binary python3.7 --with-versions

ENV PYTHON3_PREFIX /usr
ENV PYTHON3_VERSION python3.7

