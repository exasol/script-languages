FROM {{base_test_deps}}

RUN mkdir /conf /buckets

RUN ldconfig

RUN mkdir /exaudfclient /exaudf
COPY /exaudfclient/ /exaudfclient/

WORKDIR /exaudfclient/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/lib/
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$MAMBA_ROOT_PREFIX/include/
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:$MAMBA_ROOT_PREFIX/include/
RUN ./build.sh --config no-tty -c dbg --config python --config test-binaries
RUN cp -r -L bazel-bin/* /exaudf

WORKDIR /exaudfclient/base
RUN ./test_udfclient.sh /exaudf/exaudfclient
RUN ./test_udfclient.sh /exaudf/exaudfclient_static

WORKDIR /
RUN mkdir /exasol_emulator
COPY emulator/ /exasol_emulator
COPY /exaudfclient/base/exaudflib/zmqcontainer.proto /exasol_emulator
RUN cd /exasol_emulator && protoc zmqcontainer.proto --python_out=.
