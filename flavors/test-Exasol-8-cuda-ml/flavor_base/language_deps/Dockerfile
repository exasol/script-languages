FROM {{udfclient_deps}}

RUN mkdir -p /build_info/packages/language_deps

ENV CONDA_OVERRIDE_CUDA=12.9

COPY language_deps_packages.yaml /build_info/packages/language_deps_packages.yaml
RUN exaslpm install --package-file /build_info/packages/language_deps_packages.yaml --build-step language_deps

ENV PYTHON3_PREFIX=$MAMBA_ROOT_PREFIX
ENV PYTHON3_VERSION=python3.12

