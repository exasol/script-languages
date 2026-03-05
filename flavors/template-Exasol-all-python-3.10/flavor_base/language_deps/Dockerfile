FROM {{udfclient_deps}}

RUN mkdir -p /build_info/packages/language_deps

COPY language_deps_packages.yml /build_info/packages/language_deps_packages.yml
RUN exaslpm install --package-file /build_info/packages/language_deps_packages.yml --build-step language_deps

RUN /scripts/install_scripts/install_python3.10_pip.sh "pip == 21.3.1"


ENV PYTHON3_PREFIX /usr
ENV PYTHON3_VERSION python3.10

