FROM {{ nvidia_driver_deps }}

ENV ENV_NAME="base"
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ENV MAMBA_DOCKERFILE_ACTIVATE=1

COPY conda_deps/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh

SHELL ["/bin/bash", "-l", "-c"]
ENV MICROMAMBA_VERSION=0.24.0
RUN /scripts/install_scripts/install_micromamba.sh "$MICROMAMBA_VERSION"

RUN addgroup --gid 1000 exasolution
RUN adduser --disabled-login --uid 1000 --gid 1000 exasolution --gecos "First Last,RoomNumber,WorkPhone,HomePhone"
RUN addgroup --gid 500 exausers 
RUN adduser --disabled-login --uid 500 --gid 500 exadefusr --gecos "First Last,RoomNumber,WorkPhone,HomePhone"

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

