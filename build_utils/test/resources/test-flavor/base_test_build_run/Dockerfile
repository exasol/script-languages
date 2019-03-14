FROM {{base_test_deps}}

RUN mkdir /exaudf_src /exaudf
COPY src/ /exaudf_src/

ENV VERBOSE_BUILD "--subcommands --verbose_failures"

RUN cd /exaudf_src/ && \
	bash build.sh -c dbg --define benchmark=true //:exaudfclient && \
	cp -r -L bazel-bin/* /exaudf

RUN mkdir /exasol_emulator
COPY emulator/ /exasol_emulator
COPY src/exaudflib/zmqcontainer.proto /exasol_emulator
RUN cd /exasol_emulator && protoc zmqcontainer.proto --python_out=.