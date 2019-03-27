FROM {{build_deps}}

RUN mkdir /exaudf_src /exaudf
COPY src/ /exaudf_src/

ENV VERBOSE_BUILD "--subcommands --verbose_failures"

RUN cd /exaudf_src/ && \
	bash build.sh -c dbg --define python=true //:exaudfclient && \
	cp -r -L bazel-bin/* /exaudf