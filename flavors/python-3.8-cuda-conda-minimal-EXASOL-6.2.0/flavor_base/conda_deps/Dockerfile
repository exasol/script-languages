FROM {{nvidia_driver_deps}}

ENV ENV_NAME="base"
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY conda_deps/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh

ARG MAMBA_DOCKERFILE_ACTIVATE=1

SHELL ["/bin/bash", "-l", "-c"]
RUN /scripts/install_scripts/install_micromamba.sh 0.23.0

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

