FROM {{language_deps}}

RUN mkdir -p /build_info/packages

COPY build_deps_packages.yml /build_info/packages/build_deps_packages.yml
RUN exaslpm install --package-file /build_info/packages/build_deps_packages.yml --build-step build_deps

RUN exaslpm export-variables --out-file /env

RUN curl -fsSL -o swig-2.0.4.tar.gz \
        https://exasol-script-languages-dependencies.s3.eu-central-1.amazonaws.com/swig-2.0.4.tar.gz && \
    tar zxf swig-2.0.4.tar.gz && \
    (cd swig-2.0.4 && source /env && ./configure --build=$SWIG_BUILD_ARCH-unknown-linux-gnu --host=$HOST_PLATFORM-unknown-linux-gnu --prefix=/usr && make -j"$(nproc)" && make install) && \
    rm -rf swig-2.0.4 swig-2.0.4.tar.gz
