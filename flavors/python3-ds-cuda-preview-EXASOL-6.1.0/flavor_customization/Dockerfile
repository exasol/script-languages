FROM {{flavor_base_deps}}

RUN mkdir -p /build_info/packages/flavor_customization

COPY flavor_customization/packages/apt_get_packages /build_info/packages/flavor_customization
RUN /scripts/install_via_apt.pl --file /build_info/packages/flavor_customization/apt_get_packages --with-versions

COPY flavor_customization/packages/pip3_packages /build_info/packages/flavor_customization
RUN /scripts/install_via_pip.pl --file /build_info/packages/flavor_customization/pip3_packages --python-binary python3


##########################################################################
##########################################################################
# Below here you can add any Dockerfile commands except FROM.            
# However only commands which change the filesystem will be              
# reflected to the final script-language container.                      
# We recommand the use of onyl RUN, COPY and ADD. For more               
# information about Dockerfile commands please check the reference       
# https://docs.docker.com/engine/reference/builder/#dockerfile-reference 
##########################################################################
##########################################################################


# Examples:

# With RUN you can run any command you usally would run in bash shell.
# --------------------------------------------------------------------
# RUN curl -o data-science-examples.zip https://codeload.github.com/exasol/data-science-examples/zip/master
# RUN unzip data-science-examples.zip
# RUN git clone https://github.com/exasol/data-science-examples

# Copy a file from ./ into the image. 
# In case of exaslct it is import to prefix files with flavor_customization if you want to copy from ./ 
# -----------------------------------------------------------------------------------------------------
# COPY flavor_customization/packages/pip3_packages /build_info/packages/flavor_base_deps
