FROM {{build_deps}}

RUN mkdir /exaudfclient /exaudf
COPY exaudfclient/base /exaudfclient/base

WORKDIR /exaudfclient/base
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/lib/
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$MAMBA_ROOT_PREFIX/include/
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:$MAMBA_ROOT_PREFIX/include/
RUN ./build.sh --config no-tty --config optimize --config python --config fast-binary-py3
RUN cp -r -L bazel-bin/* /exaudf

RUN ./test_udfclient.sh /exaudf/exaudfclient_py3

WORKDIR /exaudf

RUN rm -r /exaudfclient

COPY --from={{build_deps}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965


RUN mkdir /conf /buckets
