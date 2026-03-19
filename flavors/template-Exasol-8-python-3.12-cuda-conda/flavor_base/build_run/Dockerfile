FROM {{build_deps}}

RUN mkdir /exaudfclient /exaudf
COPY exaudfclient/ /exaudfclient/

RUN exaslpm export-variables --out-file /env

WORKDIR /exaudfclient/

RUN bash -c "source /env && ./build.sh --config no-tty --config optimize --config python --config slow-wrapper --config fix_conda_ar_tool"
RUN cp -r -L bazel-bin/* /exaudf

WORKDIR /exaudfclient/base
RUN ./test_udfclient.sh /exaudf/exaudfclient

WORKDIR /exaudf

RUN rm -r /exaudfclient

RUN mkdir /conf /buckets
