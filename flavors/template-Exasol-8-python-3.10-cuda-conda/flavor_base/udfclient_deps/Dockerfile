FROM {{ conda_deps }}


COPY udfclient_deps_packages.yml /build_info/packages/udfclient_deps_packages.yml
RUN exaslpm install --package-file /build_info/packages/udfclient_deps_packages.yml --build-step udfclient_deps
