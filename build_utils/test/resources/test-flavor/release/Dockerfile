FROM ubuntu:18.04
RUN mkdir /conf /buckets

COPY --from={{build_run}} /usr /usr
COPY --from={{build_run}} /lib /lib
COPY --from={{build_run}} /bin /bin
COPY --from={{build_run}} /opt /opt
COPY --from={{build_run}} /etc /etc

COPY --from={{flavor_customization}} /usr /usr
COPY --from={{flavor_customization}} /lib /lib
COPY --from={{flavor_customization}} /bin /bin
COPY --from={{flavor_customization}} /opt /opt
COPY --from={{flavor_customization}} /etc /etc

RUN ldconfig

COPY --from={{build_run}} /exaudf /exaudf

