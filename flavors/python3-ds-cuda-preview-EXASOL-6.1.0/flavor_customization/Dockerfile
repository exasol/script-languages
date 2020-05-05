############################################################################################
############################################################################################
# This Dockerfile allows you to extend this flavor by installing packages or adding files. 
# IF you didn't change the lines below, you can add packages and their version to the  
# files in ./packages and they get automatically installed.                                
############################################################################################
############################################################################################

#######################################################################
#######################################################################
# Do not change the following lines unless you know what you are doing 
#######################################################################
#######################################################################

FROM {{flavor_base_deps}}

RUN mkdir -p /build_info/packages/flavor_customization

COPY flavor_customization/packages/apt_get_packages /build_info/packages/flavor_customization
RUN /scripts/install_via_apt.pl --file /build_info/packages/flavor_customization/apt_get_packages --with-versions

COPY flavor_customization/packages/pip3_packages /build_info/packages/flavor_customization
RUN /scripts/install_via_pip.pl --file /build_info/packages/flavor_customization/pip3_packages --python-binary python3 --with-versions


##########################################################################
##########################################################################
# Below this text you can add any Dockerfile commands except of FROM.            
# However only commands which change the filesystem will be              
# reflected to the final script-language container.                      
# We recommand to use of onyl RUN, COPY and ADD. For more               
# information about Dockerfile commands please check the reference       
# https://docs.docker.com/engine/reference/builder/#dockerfile-reference 
##########################################################################
##########################################################################


# Examples:

# With RUN you can run any command you usally would run in a bash shell.
# --------------------------------------------------------------------
# RUN curl -o data-science-examples.zip https://codeload.github.com/exasol/data-science-examples/zip/master
# RUN unzip data-science-examples.zip
# RUN git clone https://github.com/exasol/data-science-examples

# Copy a file from ./ into the image. 
# In case of exaslct it is important to prefix source files or directories with flavor_customization 
# if you want to copy files or directories located in the same directory as this Dockerfile
# -----------------------------------------------------------------------------------------------------
# COPY flavor_customization/packages/pip3_packages /build_info/packages/flavor_base_deps
