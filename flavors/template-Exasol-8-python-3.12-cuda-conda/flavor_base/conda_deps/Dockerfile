FROM exasol/script-language-container:exaslpm-1.1.0-ubuntu-24.04
ENV DEBIAN_FRONTEND=noninteractive

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc

COPY scripts /scripts

RUN mkdir -p /build_info/packages
COPY conda_deps_packages.yaml /build_info/packages/conda_deps_packages.yaml
RUN exaslpm install --package-file /build_info/packages/conda_deps_packages.yaml --build-step conda_deps



RUN usermod -l exasolution ubuntu
RUN groupmod -n exasolution ubuntu
RUN addgroup --gid 500 exausers
RUN adduser --disabled-login --uid 500 --gid 500 exadefusr --gecos "First Last,RoomNumber,WorkPhone,HomePhone"


ENV MAMBA_EXE="/bin/micromamba"
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME="base"
ENV MAMBA_ROOT_PREFIX="/opt/conda"

COPY scripts/install_scripts/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh

RUN /scripts/install_scripts/prepare_conda_env.sh

SHELL ["/bin/bash", "-l", "-c"]

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
