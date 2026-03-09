FROM {{ conda_deps }}


COPY udfclient_deps_packages.yaml /build_info/packages/udfclient_deps_packages.yaml
RUN exaslpm install --package-file /build_info/packages/udfclient_deps_packages.yaml --build-step udfclient_deps
