FROM ubuntu:20.04

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt-get update && apt-get install --no-install-recommends -y python3-distutils python-distutils-extra r-base perl-base perl-doc locales curl git build-essential
COPY install_scripts/install_python*_pip.sh / 
RUN /install_python3.8_pip.sh "pip>20"
COPY . /scripts
