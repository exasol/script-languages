FROM {{udfclient_deps}}

RUN apt-get -y update && \
    apt-get -y install \
        python-dev  \
        python-pip \
        python-pandas \
        libpcre3-dev && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

ENV PYTHON2_PREFIX /usr
ENV PYTHON2_VERSION python2.7