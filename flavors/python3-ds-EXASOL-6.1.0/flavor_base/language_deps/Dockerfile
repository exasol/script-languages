FROM {{udfclient_deps}}

RUN mkdir -p /build_info/packages/language_deps
COPY language_deps/packages/apt_get_packages /build_info/packages/language_deps

RUN /scripts/install_scripts/install_via_apt.pl --file /build_info/packages/language_deps/apt_get_packages

RUN curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py

COPY language_deps/packages/pip3_packages /build_info/packages/language_deps
RUN /scripts/install_scripts/install_via_pip.pl --file /build_info/packages/language_deps/pip3_packages --python-binary python3

ENV PYTHON3_PREFIX /usr
ENV PYTHON3_VERSION python3.6

