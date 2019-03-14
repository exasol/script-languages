FROM {{flavor_base_deps}}

RUN apt-get -y update && \
    apt-get -y install \
        python-sklearn && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig