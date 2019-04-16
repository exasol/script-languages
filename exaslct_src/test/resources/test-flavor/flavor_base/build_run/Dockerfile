FROM {{language_deps}}

COPY --from={{build_deps}} /usr /usr
COPY --from={{build_deps}} /lib /lib
COPY --from={{build_deps}} /bin /bin
COPY --from={{build_deps}} /opt /opt
COPY --from={{build_deps}} /etc /etc

RUN ldconfig

RUN mkdir /exaudf_src /exaudf
COPY src/ /exaudf_src/

#ENV VERBOSE_BUILD "--subcommands --verbose_failures"

RUN cd /exaudf_src/ && \
	bash build.sh -c dbg --define python=true //:exaudfclient && \
	cp -r -L bazel-bin/* /exaudf