FROM {{build_deps}}

RUN mkdir /exaudfclient /exaudf
COPY exaudfclient/ /exaudfclient/

WORKDIR /exaudfclient/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/lib/
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$MAMBA_ROOT_PREFIX/include/
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:$MAMBA_ROOT_PREFIX/include/
RUN ./build.sh --config no-tty --config optimize --config python --config slow-wrapper  --action_env=AR=x86_64-conda-linux-gnu-gcc-ar --features=-archive_param_file
RUN cp -r -L bazel-bin/* /exaudf

WORKDIR /exaudfclient/base
RUN ./test_udfclient.sh /exaudf/exaudfclient

WORKDIR /exaudf

RUN rm -r /exaudfclient

COPY --from={{build_deps}} /build_info /build_info
RUN true # workaround for https://github.com/moby/moby/issues/37965


RUN mkdir /conf /buckets
