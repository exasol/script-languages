FROM {{udfclient_deps}}

RUN mkdir -p /packages
COPY language_deps/packages /packages/language_deps

RUN apt-get -y update && \
    apt-get -y install $(cat /packages/language_deps/apt_get_packages) && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

ENV PYTHON2_PREFIX /usr
ENV PYTHON2_VERSION python2.7
ENV JAVA_PREFIX /usr/lib/jvm/java-9-openjdk-amd64

