FROM {{build_deps}}

RUN mkdir -p /packages
COPY base_test_deps/packages /packages/base_test_deps

RUN apt-get -y update && \
    apt-get -y install $(cat /packages/base_test_deps/apt_get_packages) && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig