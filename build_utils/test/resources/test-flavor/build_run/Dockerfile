FROM {{build_deps}}

RUN mkdir /exaudf_src /exaudf
COPY src/ /exaudf_src/

ENV PROTOBUF_BIN=/usr/bin/protoc
ENV PROTOBUF_LIBRARY_PREFIX=/usr/lib/x86_64-linux-gnu
ENV PROTOBUF_INCLUDE_PREFIX=/usr/include/google/protobuf/
ENV VERBOSE_BUILD "--subcommands --verbose_failures"

RUN cd /exaudf_src/ && \
	bash build.sh -c opt --define benchmark=true //:exaudfclient && \
	cp -r -L bazel-bin/* /exaudf