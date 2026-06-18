FROM {{flavor_customization}}
RUN mkdir /conf /buckets

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
    exaslpm list-all-installed-packages --out-file /build_info/actual_installed_packages/release/packages.yml
