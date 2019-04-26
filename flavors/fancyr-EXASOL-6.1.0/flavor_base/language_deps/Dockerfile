FROM {{udfclient_deps}}

RUN mkdir -p /packages
COPY language_deps/packages /packages/language_deps

# ppa:marutter/rrutter3.5 needed for R 3.5
RUN apt-get update && \
    gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    gpg -a --export E298A3A825C0D65DFD57CBB651716619E084DAB9 | apt-key add -  && \
    add-apt-repository -y 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
    apt-get -y update && \
    apt-get -y install $(cat /packages/language_deps/apt_get_packages) && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig