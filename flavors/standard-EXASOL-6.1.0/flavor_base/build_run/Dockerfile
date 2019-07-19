FROM {{language_deps}}

COPY --from={{build_deps}} /usr /usr
COPY --from={{build_deps}} /lib /lib
COPY --from={{build_deps}} /bin /bin
COPY --from={{build_deps}} /opt /opt
COPY --from={{build_deps}} /etc /etc
COPY --from={{build_deps}} /env /env

RUN ldconfig

RUN mkdir /exaudf_src /exaudf /exaudf/src 
COPY src/ /exaudf_src/

WORKDIR /exaudf_src/
RUN ["/bin/bash", "-c", "source /env && bash build.sh --config no-tty --config optimize --config python --config java --config optimize-r --config fast-binary-both"]
RUN cp -r -L bazel-bin/* /exaudf

# These source files is required for the building of the protegrity client
RUN cp debug_message.h exaudfclient.cc exaudflib/exaudflib.cc exaudflib/exaudflib.h exaudflib/zmqcontainer.proto /exaudf/src

RUN ./test_udfclient.sh /exaudf/exaudfclient
RUN ./test_udfclient.sh /exaudf/exaudfclient_py3

COPY --from={{build_deps}} /build_info /build_info