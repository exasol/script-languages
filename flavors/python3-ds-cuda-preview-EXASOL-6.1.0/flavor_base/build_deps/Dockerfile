FROM {{cuda_deps}}
ENV DEBIAN_FRONTEND=noninteractive

ENV ARCHIVE_UBUNTU_PREFIX=""
RUN sed --in-place --regexp-extended "s/(\/\/)(archive\.ubuntu)/\1$ARCHIVE_UBUNTU_PREFIX\2/" /etc/apt/sources.list

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc
RUN mkdir -p /packages
COPY build_deps/packages /packages/build_deps

RUN apt-get -y update && \
    apt-get install -y $(cat /packages/build_deps/apt_get_packages) && \
	echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
	curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
	apt-get update && \
	apt-get install -y bazel && \
	curl -L -o swig-2.0.4.tar.gz prdownloads.sourceforge.net/swig/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    (cd swig-2.0.4 && ./configure --prefix=/usr && make && make install) && \
    rm -rf swig-2.0.4 swig-2.0.4.tar.gz && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

RUN touch /env && \
    echo "export PROTOBUF_BIN=/usr/bin/protoc" >> /env && \
    echo "export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64" >> /env