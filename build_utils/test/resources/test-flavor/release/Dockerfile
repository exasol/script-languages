FROM ubuntu:18.04

COPY --from={{build_run}} /usr /usr
COPY --from={{build_run}} /opt /opt
COPY --from={{flavor_customization}} /usr /usr
COPY --from={{flavor_customization}} /opt /opt
COPY --from={{build_run}} /exaudf /exaudf

RUN mkdir /conf /buckets