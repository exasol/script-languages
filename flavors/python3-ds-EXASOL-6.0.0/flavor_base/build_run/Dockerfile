FROM {{language_deps}}

COPY --from={{build_deps}} /usr /usr
COPY --from={{build_deps}} /lib /lib
COPY --from={{build_deps}} /bin /bin
COPY --from={{build_deps}} /opt /opt
COPY --from={{build_deps}} /etc /etc
COPY --from={{build_deps}} /env /env

RUN ldconfig

RUN mkdir /exaudfclient /exaudf
COPY exaudfclient/base /exaudfclient/base

WORKDIR /exaudfclient/base
RUN ["/bin/bash", "-c", "source /env && bash build.sh --config no-tty --config optimize --config python --config fast-binary-py3"]
RUN cp -r -L bazel-bin/* /exaudf

RUN ./test_udfclient.sh /exaudf/exaudfclient_py3

WORKDIR /exaudf

RUN rm -r /exaudfclient

COPY --from={{build_deps}} /build_info /build_info
