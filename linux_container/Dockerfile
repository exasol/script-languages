FROM ubuntu:16.04

#RUN echo nameserver 172.17.0.1 > /etc/resolv.conf

COPY ext/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc 
RUN apt-get update && \
    apt-get -y install \
        software-properties-common \
        coreutils \
        locales \
        openjdk-8-jdk-headless \
        python2.7-dev  \
        r-base \
        r-base-dev \
        protobuf-compiler \
        libprotobuf-dev \
        scons \
        libssl-dev \
        wget \
        libpcre3-dev \
        swig2.0 \
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
        python-tz \
        python-tzlocal \
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
        r-cran-randomforest \
	r-cran-bradleyterry2 \
	r-cran-brglm \
	r-cran-car \
	r-cran-caret \
	r-cran-colorspace \
	r-cran-dichromat \
	r-cran-digest \
	r-cran-foreach \
	r-cran-ggplot2 \
	r-cran-gtools \
	r-cran-iterators \
	r-cran-labeling \
	r-cran-lme4 \
	r-cran-magrittr \
	r-cran-minqa \
	r-cran-munsell \
	r-cran-nloptr \
	r-cran-plyr \
	r-cran-profilemodel \
	r-cran-proto \
	r-cran-rcpp \
	r-cran-rcppeigen \
	r-cran-reshape2 \
	r-cran-scales \
	r-cran-stringr \
        maven \
        git && \
    pip3 install protobuf && \
    Rscript -e 'install.packages("rredis",repos = "http://cran.r-project.org")' && \
    Rscript -e 'install.packages("data.table",repos = "http://cran.r-project.org")' && \
    Rscript -e 'install.packages("htmltools",repos = "http://cran.r-project.org")' && \
    Rscript -e 'install.packages("flashClust",repos = "http://cran.r-project.org")' && \
    locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    echo /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server >> /etc/ld.so.conf && \
    (apt install -y wget cmake g++ && \
     mkdir /tmp/build_zmq && \
     cd /tmp/build_zmq && \
     wget https://github.com/zeromq/libzmq/archive/v4.2.3.zip && \
     unzip v4.2.3.zip && \
     cd libzmq-4.2.3 && \
     mkdir build && cd build && \
     cmake .. && \
     make install -j4 && \
     cd ../../ && rm -rf * && \
     wget https://github.com/zeromq/cppzmq/archive/v4.2.3.zip && \
     unzip v4.2.3.zip && \
     cd cppzmq-4.2.3 && \
     mkdir build && cd build && \
     cmake .. && make install -j4 && \
     cd ../../../ && rm -rf build_zmq) && \
     apt-get install -y python-zmq python3-zmq && \
    ldconfig
RUN addgroup --gid 1000 exasolution
RUN adduser --disabled-login --uid 1000 --gid 1000 exasolution

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  


#    wget prdownloads.sourceforge.net/swig/swig-2.0.4.tar.gz && \
#    tar zxf swig-2.0.4.tar.gz && \
#    (cd swig-2.0.4 && ./configure --prefix=/usr && make && make install) && \
#    rm -rf swig-2.0.4 swig-2.0.4.tar.gz && \
#
