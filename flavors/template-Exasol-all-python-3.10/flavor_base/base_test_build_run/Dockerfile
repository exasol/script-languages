FROM {{language_deps}}

RUN mkdir /conf /buckets

COPY --from={{base_test_deps}} /usr /usr
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{base_test_deps}} /lib /lib
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{base_test_deps}} /bin /bin
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{base_test_deps}} /opt /opt
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{base_test_deps}} /etc /etc
RUN true # workaround for https://github.com/moby/moby/issues/37965

COPY --from={{base_test_deps}} /env /env
RUN true # workaround for https://github.com/moby/moby/issues/37965


RUN ldconfig

RUN mkdir /exaudfclient /exaudf
COPY /exaudfclient/base /exaudfclient/base

WORKDIR /exaudfclient/base
RUN ["/bin/bash", "-c", "source /env && bash build.sh --strip never --config no-tty -c dbg --config python --config slow-wrapper-py3"]
RUN cp -r -L bazel-bin/* /exaudf
RUN ./test_udfclient.sh /exaudf/exaudfclient_py3
#RUN ./test_udfclient.sh /exaudf/exaudfclient_py3_static

WORKDIR /
RUN mkdir /exasol_emulator
COPY emulator/ /exasol_emulator
COPY /exaudfclient/base/exaudflib/zmqcontainer.proto /exasol_emulator
RUN cd /exasol_emulator && protoc zmqcontainer.proto --python_out=.
