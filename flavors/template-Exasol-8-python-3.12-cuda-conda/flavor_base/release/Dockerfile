FROM {{flavor_customization}}

ENV ENV_NAME="base"
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

RUN mkdir -p /conf /buckets /nvidia/driver /nvidia/utils

COPY --from={{flavor_base_deps}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_base_deps}} /scripts /scripts
RUN true # workaround for https://github.com/moby/moby/issues/37965

RUN ldconfig

COPY --from={{build_run}} /exaudf /exaudf
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{build_run}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965


RUN mkdir -p /build_info/actual_installed_packages/release && \
    /scripts/list_installed_scripts/list_installed_apt.sh > /build_info/actual_installed_packages/release/apt_get_packages && \
    /scripts/list_installed_scripts/list_installed_pip.sh "$MAMBA_ROOT_PREFIX/bin/python3.12" > /build_info/actual_installed_packages/release/python3_pip_packages && \
   /scripts/list_installed_scripts/list_installed_conda.sh > /build_info/actual_installed_packages/release/conda_packages
