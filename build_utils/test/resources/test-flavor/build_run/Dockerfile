FROM {{build_deps}}

RUN mkdir /exaudf_src /exaudf
COPY src/ /exaudf_src/

ENV VERBOSE_BUILD "--subcommands --verbose_failures"

RUN cd /exaudf_src/ && \
	bash build.sh -c opt --define benchmark=true //:exaudfclient && \
	cp -r -L bazel-bin/* /exaudf