FROM {{udfclient_deps}}

RUN mkdir -p /build_info/packages/language_deps

COPY language_deps_packages.yaml /build_info/packages/language_deps_packages.yaml
RUN exaslpm install --package-file /build_info/packages/language_deps_packages.yaml --build-step language_deps

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 #Needed for Bazel build
