FROM {{ language_deps }}

RUN mkdir -p /build_info/packages
COPY flavor_base_deps/packages/apt_get_packages /build_info/packages/flavor_base_deps

RUN apt-get -y update && \
    apt-get -y install $(cat /build_info/packages/flavor_base_deps/apt_get_packages) && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig


COPY flavor_base_deps/packages/python3_pip_packages /build_info/packages/flavor_base_deps
RUN /scripts/install_scripts/install_via_pip.pl --file /build_info/packages/flavor_base_deps/python3_pip_packeges --python-binary python3
