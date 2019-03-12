FROM {{language_deps}}

RUN apt-get -y update && \
    apt-get -y install \
        python-pip \
        python-lxml \
        python-cjson \
        python-pandas \
        python-requests && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig