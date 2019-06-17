FROM {{ language_deps }}

RUN mkdir -p /build_info/packages
COPY flavor_base_deps/packages /build_info/packages/flavor_base_deps

RUN apt-get -y update && \
    apt-get -y install $(cat /build_info/packages/flavor_base_deps/apt_get_packages) && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

RUN /scripts/install_packages_batch.sh "PYTHON3-PIP" "pip install --ignore-installed <<list>>" "'<<package>>'" " " /build_info/packages/flavor_base_deps/pip3_packages