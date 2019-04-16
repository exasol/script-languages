FROM {{udfclient_deps}}

COPY language_deps/apt_get_packages /language_deps_apt_get_packages

RUN apt-get -y update && \
    apt-get -y install $(cat /language_deps_apt_get_packages) && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

ENV PYTHON2_PREFIX /usr
ENV PYTHON2_VERSION python2.7
ENV PYTHON3_PREFIX /usr
ENV PYTHON3_VERSION python3.6
ENV JAVA_PREFIX /usr/lib/jvm/java-11-openjdk-amd64

