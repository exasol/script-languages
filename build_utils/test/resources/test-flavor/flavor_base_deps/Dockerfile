FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

COPY ext/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

RUN apt-get -y update && \
    apt-get -y install \
        curl \
        locales \
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