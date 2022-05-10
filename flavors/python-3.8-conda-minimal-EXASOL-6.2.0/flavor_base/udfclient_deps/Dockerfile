FROM {{conda_deps}}

COPY udfclient_deps/packages /build_info/packages/udfclient_deps
RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/udfclient_deps/apt_get_packages --with-versions

ENV PROTOBUF_LIBRARY_PREFIX=/usr/lib/x86_64-linux-gnu
ENV PROTOBUF_INCLUDE_PREFIX=/usr/include/google/protobuf/
