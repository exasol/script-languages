FROM {{language_deps}}

COPY --from={{base_test_deps}} /usr /usr
COPY --from={{base_test_deps}} /lib /lib
COPY --from={{base_test_deps}} /bin /bin
COPY --from={{base_test_deps}} /opt /opt
COPY --from={{base_test_deps}} /etc /etc
COPY --from={{base_test_deps}} /env /env

RUN ldconfig

RUN mkdir /exaudf_src /exaudf
COPY src/ /exaudf_src/

#ENV VERBOSE_BUILD "--subcommands --verbose_failures"

WORKDIR /exaudf_src/
RUN ["/bin/bash", "-c", "source /env && bash build.sh -c dbg --define python=true //:exaudfclient_py3"]
RUN cp -r -L bazel-bin/* /exaudf

RUN mkdir /exasol_emulator
COPY emulator/ /exasol_emulator
COPY src/exaudflib/zmqcontainer.proto /exasol_emulator
RUN cd /exasol_emulator && protoc zmqcontainer.proto --python_out=.