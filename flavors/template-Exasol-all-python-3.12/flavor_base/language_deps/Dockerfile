FROM {{udfclient_deps}}

RUN mkdir -p /build_info/packages/language_deps

COPY language_deps_packages.yml /build_info/packages/language_deps_packages.yml
RUN exaslpm install --package-file /build_info/packages/language_deps_packages.yml --build-step language_deps

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 #Needed for Bazel build
