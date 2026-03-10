FROM {{build_deps}}

RUN mkdir -p /build_info/packages

COPY base_test_deps_packages.yaml /build_info/packages/base_test_deps_packages.yaml
RUN exaslpm install --package-file /build_info/packages/base_test_deps_packages.yaml --build-step base_test_deps
