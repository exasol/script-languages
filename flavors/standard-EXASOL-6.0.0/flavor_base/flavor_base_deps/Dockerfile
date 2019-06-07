FROM {{ language_deps }}

RUN mkdir -p /build_info/packages/flavor_base_deps
COPY flavor_base_deps/packages/apt_get_packages /build_info/packages/flavor_base_deps

RUN apt-get -y update && \
    apt-get -y install $(cat /build_info/packages/flavor_base_deps/apt_get_packages) && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

COPY scripts /scripts
COPY flavor_base_deps/packages/cran_packages /build_info/packages/flavor_base_deps
RUN /scripts/install_packages_batch.sh "CRAN" "Rscript -e 'install.packages(c(<<list>>),repos=\"http://cran.r-project.org\")'" '"<<package>>"' "," /build_info/packages/flavor_base_deps/cran_packages
COPY flavor_base_deps/packages/pip_packages /build_info/packages/flavor_base_deps
RUN /scripts/install_packages_batch.sh "PYTHON2-PIP" "pip install <<list>>" "<<package>>" " " /build_info/packages/flavor_base_deps/pip_packages
COPY flavor_base_deps/packages/pip3_packages /build_info/packages/flavor_base_deps
RUN /scripts/install_packages_batch.sh "PYTHON3-PIP3" "pip3 install <<list>>" "<<package>>" " " /build_info/packages/flavor_base_deps/pip3_packages