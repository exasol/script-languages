FROM ubuntu:16.04


COPY ext/01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc 

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    #add-apt-repository ppa:webupd8team/java -y && \
    #apt-get update && \
    #echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
    #apt-get install -y oracle-java8-installer && \
    #apt-get clean && \
    #apt-get -y autoremove && \
    #rm -rf /var/cache/oracle-jdk8-installer && \
    ## uname, etc: \
    apt-get -y install coreutils && \
    apt-get -y install openjdk-8-jdk-headless && \
    apt-get -y install python2.7-dev  && \
    apt-get -y install r-base  && \
    apt-get -y install r-base-dev && \
    apt-get -y install libzmq-dev && \
    apt-get -y install python-zmq  && \
    apt-get -y install python3-zmq  && \
    apt-get -y install protobuf-compiler && \ 
    apt-get -y install libprotobuf-dev && \
    apt-get -y install scons && \
    apt-get -y install libssl-dev && \
    apt-get -y install wget && \
    apt-get -y install libpcre3-dev && \
    wget prdownloads.sourceforge.net/swig/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    cd swig-2.0.4 && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf swig-2.0.4 && rm swig-2.0.4.tar.gz && \
    apt-get -y install python-cffi && \
    apt-get -y install python-cryptography && \
    apt-get -y install python-docutils && \
    apt-get -y install python-enum34 && \
    apt-get -y install python-idna && \
    apt-get -y install python-ipaddress && \
    apt-get -y install ipython && \
    apt-get -y install python-jinja2 && \
    apt-get -y install python-martian && \
    apt-get -y install python-pip && \
    apt-get -y install python3-pip && \
    apt-get -y install python-protobuf && \
    pip3 install protobuf && \
    apt-get -y install python-pyasn1 && \
    apt-get -y install python-pyasn1-modules && \
    apt-get -y install python-pycparser && \
    apt-get -y install python-pycryptopp && \
    apt-get -y install pyflakes && \
    apt-get -y install python-pyftpdlib && \
    apt-get -y install python-pygments && \
    apt-get -y install python-pykickstart && \
    apt-get -y install python-pyodbc && \
    apt-get -y install python-openssl && \
    apt-get -y install python-pypdf && \
    apt-get -y install python-ldb && \
    apt-get -y install python-ldap && \
    apt-get -y install cython && \
    apt-get -y install python-roman && \
    apt-get -y install python-samba && \
    apt-get -y install python-sklearn && \
    apt-get -y install python-talloc && \
    apt-get -y install python-cjson && \
    apt-get -y install python-lxml && \
    apt-get -y install python-numpy && \
    apt-get -y install python-setuptools && \
    apt-get -y install python-pandas && \
    easy_install pytz && \
    apt-get -y install python-redis && \
    apt-get -y install python-scipy && \
    apt-get -y install r-cran-rcurl && \
    apt-get -y install r-cran-xml && \
    apt-get -y install r-cran-formula && \
    apt-get -y install r-cran-rcolorbrewer && \
    apt-get -y install r-cran-rodbc && \
    apt-get -y install r-cran-acepack && \
    apt-get -y install r-cran-chron && \
    apt-get -y install r-cran-e1071 && \
    apt-get -y install r-cran-fastcluster && \
    apt-get -y install r-cran-gbm && \
    apt-get -y install r-cran-gridextra && \
    apt-get -y install r-cran-gtable && \
    apt-get -y install r-cran-latticeextra && \
    apt-get -y install r-cran-randomforest && \
    Rscript -e 'install.packages("rredis",repos = "http://cran.r-project.org")' && \
    # Generate and set the locale \
    locale-gen en_US.UTF-8  && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get clean && \
    apt-get -y autoremove && \
    echo /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server >> /etc/ld.so.conf && \
    ldconfig
    #rm -rf /var/lib/apt/lists/* && \
    #rm -f /usr/lib/jvm/java-8-oracle/src.zip

RUN addgroup --gid 1000 exasolution
RUN adduser --disabled-login --uid 1000 --gid 1000 exasolution

ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  



