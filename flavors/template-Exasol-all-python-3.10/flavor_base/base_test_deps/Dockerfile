FROM {{build_deps}}

RUN mkdir -p /build_info/packages


COPY base_test_deps_packages.yml /build_info/packages/base_test_deps_packages.yml
RUN exaslpm install --package-file /build_info/packages/base_test_deps_packages.yml --build-step base_test_deps
