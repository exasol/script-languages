FROM ubuntu:16.04


COPY ext/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc 

RUN apt-get -y update && \
    apt-get -y install \
        software-properties-common \
        coreutils \
        locales \
        openjdk-8-jdk-headless \
        python2.7-dev  \
        r-base \
        r-base-dev \
        libzmq-dev \
        python-zmq \
        python3-zmq \
        protobuf-compiler \
        libprotobuf-dev \
        scons \
        libssl-dev \
        wget \
        libpcre3-dev \
        python-cffi \
        python-cryptography \
        python-docutils \
        python-enum34 \
        python-idna \
        python-ipaddress \
        ipython \
        python-jinja2 \
        python-martian \
        python-pip \
        python3-pip \
        python-protobuf \
        python-pyasn1 \
        python-pyasn1-modules \
        python-pycparser \
        python-pycryptopp \
        pyflakes \
        python-pyftpdlib \
        python-pygments \
        python-pykickstart \
        python-pyodbc \
        python-openssl \
        python-pypdf \
        python-ldb \
        python-ldap \
        cython \
        python-roman \
        python-samba \
        python-sklearn \
        python-talloc \
        python-cjson \
        python-lxml \
        python-numpy \
        python-setuptools \
        python-pandas \
        python-redis \
        python-scipy \
        r-cran-rcurl \
        r-cran-xml \
        r-cran-formula \
        r-cran-rcolorbrewer \
        r-cran-rodbc \
        r-cran-acepack \
        r-cran-chron \
        r-cran-e1071 \
        r-cran-fastcluster \
        r-cran-gbm \
        r-cran-gridextra \
        r-cran-gtable \
        r-cran-latticeextra \
        r-cran-randomforest && \
    pip3 install protobuf && \
    easy_install pytz && \
    wget prdownloads.sourceforge.net/swig/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    (cd swig-2.0.4 && ./configure --prefix=/usr && make && make install) && \
    rm -rf swig-2.0.4 swig-2.0.4.tar.gz && \
    Rscript -e 'install.packages("rredis",repos = "http://cran.r-project.org")' && \
    Rscript -e 'install.packages("data.table",repos = "http://cran.r-project.org")' && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    echo /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server >> /etc/ld.so.conf && \
    ldconfig
RUN addgroup --gid 1000 exasolution
RUN adduser --disabled-login --uid 1000 --gid 1000 exasolution

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  



