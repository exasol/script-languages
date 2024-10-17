FROM {{language_deps}}

COPY --from={{build_deps}} /usr /usr
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{build_deps}} /lib /lib
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{build_deps}} /bin /bin
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{build_deps}} /opt /opt
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{build_deps}} /etc /etc
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{build_deps}} /env /env
RUN true # workaround for https://github.com/moby/moby/issues/37965


RUN ldconfig

RUN mkdir /exaudfclient /exaudf /exaudf/src
COPY /exaudfclient /exaudfclient

WORKDIR /exaudfclient/
RUN ["/bin/bash", "-c", "source /env && bash build.sh --config no-tty --config optimize --config java --config fast-binary"]
RUN cp -r -L bazel-bin/* /exaudf && rm -r /exaudf/external

WORKDIR /exaudfclient/base
RUN ./test_udfclient.sh /exaudf/exaudfclient

WORKDIR /exaudf/

RUN rm -r /exaudfclient

COPY --from={{build_deps}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965


RUN mkdir /conf /buckets
