FROM {{base_test_deps}}

RUN mkdir /conf /buckets

RUN ldconfig

RUN mkdir /exaudfclient /exaudf
COPY /exaudfclient/base /exaudfclient/base

WORKDIR /exaudfclient/base
# We need to build with optimize here, because the debug build fails to run with a runtime linking error regarding protobuf
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MAMBA_ROOT_PREFIX/lib/
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$MAMBA_ROOT_PREFIX/include/
ENV C_INCLUDE_PATH=$C_INCLUDE_PATH:$MAMBA_ROOT_PREFIX/include/
RUN ./build.sh --config no-tty -c dbg --config python --config test-binaries-py3
RUN cp -r -L bazel-bin/* /exaudf
RUN ./test_udfclient.sh /exaudf/exaudfclient_py3
RUN ./test_udfclient.sh /exaudf/exaudfclient_py3_static

WORKDIR /
RUN mkdir /exasol_emulator
COPY emulator/ /exasol_emulator
COPY /exaudfclient/base/exaudflib/zmqcontainer.proto /exasol_emulator
RUN cd /exasol_emulator && protoc zmqcontainer.proto --python_out=.
