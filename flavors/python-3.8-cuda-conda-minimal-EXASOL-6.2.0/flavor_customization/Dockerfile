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

COPY flavor_customization/packages /build_info/packages/flavor_customization
RUN /scripts/install_scripts/install_via_conda.pl --file /build_info/packages/flavor_customization/conda_packages --channel-file /build_info/packages/flavor_customization/conda_channels --with-versions --allow-no-version --conda-binary /opt/conda/bin/mamba

COPY flavor_customization/packages/python3_pip_packages /build_info/packages/flavor_customization
RUN /scripts/install_scripts/install_via_pip.pl --file /build_info/packages/flavor_customization/python3_pip_packages --python-binary /opt/conda/bin/python3.8 --with-versions --allow-no-version

##########################################################################
##########################################################################
# Below this text you can add any Dockerfile commands except of FROM.            
# However only commands which change the filesystem will be              
# reflected to the final script-language container.                      
# We recommend to use only RUN, COPY and ADD. For example, WORKDIR, USER 
# and ENV will be not carried over into the final container. For more 
# information about Dockerfile commands, please check the reference 
# https://docs.docker.com/engine/reference/builder/#dockerfile-reference 
##########################################################################
##########################################################################


###########
###########
# Examples:
###########
###########

#################################################
# Execute a command during the build of the image
#################################################

# With RUN you can excute any command you usally would run in a bash shell.
# It usually is a good idea to cleanup downloaded files or packages 
# caches in the same in the same RUN statement to keep the image size
# small. You can execute multiple shell commands in one RUN statement
# by combine them with 
# - `&&` (only execute if the previous command was successfull), 
# - `||` (only execute if the previous command was unsuccesfull)
# - `;` (execute regardless of the success of the previous command)
# To format multiple command per RUN statement you can add a line break with `\`.
# If you need to execute complex sequences of commands you should execute
# scripts which you can copy into the image.
# -------------------------------------------------------------------------------

# RUN git clone https://github.com/exasol/data-science-examples
# RUN curl -o data-science-examples.zip https://codeload.github.com/exasol/data-science-examples/zip/master && \
#     unzip data-science-examples.zip && \
#     rm data-science-examples.zip

#####################################
# Copy a file from ./ into the image. 
#####################################

# In case of exaslct it is important to prefix source files or directories with `flavor_customization`
# if you want to copy files or directories located in the same directory as this Dockerfile
# -----------------------------------------------------------------------------------------------------

# COPY flavor_customization/packages/python3_pip_packages /build_info/packages/flavor_customization
# ADD flavor_customization/code.tar.gz /code

