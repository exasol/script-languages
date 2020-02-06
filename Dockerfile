FROM ubuntu:18.04

COPY ext/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

RUN apt-get -y update && \
    apt-get -y install \
        locales \
        python3-pip \
        git \
        bash \
        curl && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

RUN pip3 install virtualenv
RUN python3 -m virtualenv --python=python3 venv

COPY . /test
RUN rm /test/Pipfile.lock || true
