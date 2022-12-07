FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

COPY ext/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

RUN apt-get -y update && \
    apt-get -y install \
        openjdk-11-jdk \
        locales \
        python3.8-venv \
        python3-pip \
        unixodbc \
        unixodbc-dev \
        curl \
        git && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

RUN mkdir -p downloads/ODBC downloads/JDBC downloads/EXAplus
RUN curl -s https://exasol-script-languages-dependencies.s3.eu-central-1.amazonaws.com/EXASOL_ODBC-7.0.11.tar.gz | tar -C downloads/ODBC --strip-components 1 -zxf -
RUN curl -s https://x-up.s3.amazonaws.com/7.x/7.1.14/EXASOL_JDBC-7.1.14.tar.gz | tar -C downloads/JDBC --strip-components 1 -zxf -
RUN curl -s https://x-up.s3.amazonaws.com/7.x/7.1.16/EXAplus-7.1.16.tar.gz | tar -C downloads/EXAplus --strip-components 1 -zxf -
ENV EXAPLUS=/downloads/EXAplus/exaplus

COPY requirements.txt requirements.txt
RUN pip3 install -r "requirements.txt"

RUN mkdir -p /dev_tools
COPY ext/scripts/dev_tools/install_dev_tools.sh /dev_tools
