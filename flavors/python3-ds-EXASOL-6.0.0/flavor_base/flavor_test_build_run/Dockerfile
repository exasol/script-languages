FROM {{base_test_build_run}}

RUN mkdir /conf /buckets

COPY --from={{flavor_customization}} /usr /usr
COPY --from={{flavor_customization}} /opt /opt
COPY --from={{flavor_customization}} /lib /lib
COPY --from={{flavor_customization}} /bin /bin
COPY --from={{flavor_customization}} /etc /etc
COPY --from={{flavor_customization}} /build_info /build_info

RUN ldconfig

