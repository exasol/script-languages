FROM {{language_deps}}

COPY --from={{build_deps}} /usr /usr
COPY --from={{build_deps}} /lib /lib
COPY --from={{build_deps}} /bin /bin
COPY --from={{build_deps}} /opt /opt
COPY --from={{build_deps}} /etc /etc
COPY --from={{build_deps}} /env /env

RUN ldconfig

RUN mkdir /exaudf_src /exaudf
COPY src/ /exaudf_src/

#ENV VERBOSE_BUILD "--subcommands --verbose_failures"

WORKDIR /exaudf_src/
RUN ["/bin/bash", "-c", "source /env && bash build.sh --config no-tty --config optimize --config optimize-r --config fast-binary-py2"]
RUN cp -r -L bazel-bin/* /exaudf

COPY --from={{build_deps}} /build_info /build_info