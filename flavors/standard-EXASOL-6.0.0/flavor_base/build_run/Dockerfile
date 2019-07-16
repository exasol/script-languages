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

ENV VERBOSE_BUILD "--subcommands --verbose_failures"

WORKDIR /exaudf_src/
RUN ["/bin/bash", "-c", "source /env && bash build.sh -c opt --config opt --define r=true --define python=true --define java=true //:exaudfclient"]
RUN cp -r -L bazel-bin/* /exaudf
RUN cp debug_message.h exaudfclient.cc exaudflib/exaudflib.cc exaudflib/exaudflib.h exaudflib/zmqcontainer.proto /exaudf/src

COPY --from={{build_deps}} /build_info /build_info