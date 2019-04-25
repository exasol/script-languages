FROM {{udfclient_deps}}

RUN mkdir -p /packages
COPY language_deps/packages /packages/language_deps

# ppa:marutter/rrutter3.5 needed for R 3.5
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository -y ppa:marutter/rrutter3.5 && \
    apt-get -y update && \
    apt-get -y install $(cat /packages/language_deps/apt_get_packages) && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig