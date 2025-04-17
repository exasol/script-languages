FROM ubuntu:22.04

ENV ENV_NAME="base"
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

RUN mkdir -p /conf /buckets /nvidia/driver /nvidia/utils

COPY --from={{language_deps}} /usr /usr
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{language_deps}} /lib /lib
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{language_deps}} /bin /bin
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{language_deps}} /opt /opt
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{language_deps}} /etc /etc
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{language_deps}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{language_deps}} /var /var
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{language_deps}} /scripts /scripts
RUN true # workaround for https://github.com/moby/moby/issues/37965


COPY --from={{flavor_base_deps}} /usr /usr
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_base_deps}} /lib /lib
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_base_deps}} /bin /bin
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_base_deps}} /opt /opt
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_base_deps}} /etc /etc
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_base_deps}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_base_deps}} /var /var
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_base_deps}} /root /root
RUN true # workaround for https://github.com/moby/moby/issues/37965

RUN ldconfig

COPY --from={{build_run}} /exaudf /exaudf
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{build_run}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY language_definitions.json /build_info/language_definitions.json
RUN true # workaround for https://github.com/moby/moby/issues/37965

RUN mkdir -p /build_info/actual_installed_packages/release && \
    /scripts/list_installed_scripts/list_installed_apt.sh > /build_info/actual_installed_packages/release/apt_get_packages && \
    /scripts/list_installed_scripts/list_installed_pip.sh "$MAMBA_ROOT_PREFIX/bin/python3.10" > /build_info/actual_installed_packages/release/python3_pip_packages && \
   /scripts/list_installed_scripts/list_installed_conda.sh > /build_info/actual_installed_packages/release/conda_packages
