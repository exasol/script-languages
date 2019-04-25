FROM {{ language_deps }}

RUN mkdir -p /packages/flavor_base_deps
COPY flavor_base_deps/packages/apt_get_packages /packages/flavor_base_deps

RUN apt-get -y update && \
    apt-get -y install $(cat /packages/flavor_base_deps/apt_get_packages) && \
	locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 && \
    apt-get -y clean && \
    apt-get -y autoremove && \
    ldconfig

COPY scripts /scripts
COPY flavor_base_deps/packages/cran_packages /packages/flavor_base_deps
RUN /scripts/install_packages_batch.sh "CRAN" "Rscript -e 'install.packages(c(<<list>>),repos=\"http://cran.r-project.org\")'" '"<<package>>"' "," /packages/flavor_base_deps/cran_packages
COPY flavor_base_deps/packages/pip_packages /packages/flavor_base_deps
RUN /scripts/install_packages_batch.sh "PYTHON2-PIP" "pip install <<list>>" "<<package>>" " " /packages/flavor_base_deps/pip_packages
COPY flavor_base_deps/packages/pip3_packages /packages/flavor_base_deps
RUN /scripts/install_packages_batch.sh "PYTHON3-PIP3" "pip3 install <<list>>" "<<package>>" " " /packages/flavor_base_deps/pip3_packages