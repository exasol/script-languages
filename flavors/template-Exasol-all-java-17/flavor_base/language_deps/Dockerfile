FROM {{udfclient_deps}}

COPY language_deps/packages /build_info/packages/language_deps

RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/language_deps/apt_get_packages --with-versions

ENV JAVA_PREFIX /usr/lib/jvm/java-17-openjdk-amd64
