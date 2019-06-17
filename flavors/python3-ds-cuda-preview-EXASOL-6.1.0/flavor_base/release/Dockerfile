FROM {{cuda_deps}}
RUN mkdir /conf /buckets

COPY --from={{language_deps}} /usr /usr
COPY --from={{language_deps}} /lib /lib
COPY --from={{language_deps}} /bin /bin
COPY --from={{language_deps}} /opt /opt
COPY --from={{language_deps}} /etc /etc
COPY --from={{language_deps}} /build_info /build_info

COPY --from={{flavor_customization}} /usr /usr
COPY --from={{flavor_customization}} /lib /lib
COPY --from={{flavor_customization}} /bin /bin
COPY --from={{flavor_customization}} /opt /opt
COPY --from={{flavor_customization}} /etc /etc
COPY --from={{flavor_customization}} /build_info /build_info

RUN ldconfig

COPY --from={{build_run}} /exaudf /exaudf
COPY --from={{build_run}} /build_info /build_info