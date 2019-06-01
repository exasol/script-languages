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

RUN R CMD javareconf
RUN /scripts/install_packages_batch.sh "CRAN" "Rscript -e 'install.packages(c(<<list>>),repos=\"http://cran.r-project.org\",Ncpus=\"2\")'" '"<<package>>"' "," /packages/flavor_base_deps/cran_packages
#TODO add build args to docker_build, in this case for number of cores
#TODO do we need the r package shiny? Probably not, it is a web app?
#TODO do we need the r package rgl, because it needs an X server