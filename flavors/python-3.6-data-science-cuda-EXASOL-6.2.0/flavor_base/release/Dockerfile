FROM {{cuda_deps}}
RUN mkdir /conf /buckets

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


COPY --from={{flavor_customization}} /usr /usr
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_customization}} /lib /lib
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_customization}} /bin /bin
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_customization}} /opt /opt
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_customization}} /etc /etc
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_customization}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{flavor_customization}} /var /var
RUN true # workaround for https://github.com/moby/moby/issues/37965


RUN ldconfig

COPY --from={{build_run}} /exaudf /exaudf
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{build_run}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965


RUN mkdir -p /build_info/actual_installed_packages/release && \
    /scripts/list_installed_scripts/list_installed_apt.sh > /build_info/actual_installed_packages/release/apt_get_packages && \
    /scripts/list_installed_scripts/list_installed_pip.sh python3.6 > /build_info/actual_installed_packages/release/python3_pip_packages
