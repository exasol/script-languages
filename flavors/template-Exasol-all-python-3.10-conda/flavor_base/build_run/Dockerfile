FROM {{build_deps}}

RUN exaslpm export-variables --out-file /env

RUN mkdir /exaudfclient /exaudf
COPY exaudfclient/ /exaudfclient/

WORKDIR /exaudfclient/

RUN source /env && ./build.sh --config no-tty --config optimize --config python --config fast-binary --config fix_conda_ar_tool
RUN cp -r -L bazel-bin/* /exaudf

WORKDIR /exaudfclient/base
RUN ./test_udfclient.sh /exaudf/exaudfclient

WORKDIR /exaudf

RUN rm -r /exaudfclient



RUN mkdir /conf /buckets
