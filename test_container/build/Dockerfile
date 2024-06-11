FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

COPY deps/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

RUN apt-get -y update && \
    apt-get -y install \
        openjdk-17-jdk \
        openjdk-11-jdk \
        maven \
        locales \
        python3.10-venv \
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

RUN update-java-alternatives --set java-1.11.0-openjdk-amd64

RUN curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh

RUN mkdir -p downloads/ODBC downloads/JDBC downloads/EXAplus
RUN curl -s https://x-up.s3.amazonaws.com/7.x/7.1.22/EXASOL_ODBC-7.1.22.tar.gz | tar -C downloads/ODBC --strip-components 1 -zxf -
RUN curl -s https://x-up.s3.amazonaws.com/7.x/7.1.14/EXASOL_JDBC-7.1.14.tar.gz | tar -C downloads/JDBC --strip-components 1 -zxf -
RUN curl -s https://exasol-script-languages-dependencies.s3.eu-central-1.amazonaws.com/EXAplus-7.0.11.tar.gz | tar -C downloads/EXAplus --strip-components 1 -zxf -
ENV EXAPLUS=/downloads/EXAplus/exaplus

COPY deps/requirements.txt requirements.txt
RUN pip3 install -r "requirements.txt"

RUN mkdir -p /dev_tools
COPY deps/install_dev_tools.sh /dev_tools
COPY deps/maven-toolchains.xml /root/.m2/toolchains.xml
