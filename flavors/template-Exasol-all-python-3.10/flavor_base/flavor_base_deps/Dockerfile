FROM {{ language_deps }}

RUN mkdir -p /build_info/packages

COPY flavor_base_deps_packages.yaml /build_info/packages/flavor_base_deps_packages.yaml
RUN exaslpm install --package-file /build_info/packages/flavor_base_deps_packages.yaml --build-step flavor_base_deps

