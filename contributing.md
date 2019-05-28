# Contributing to the Script Language Containers

## :tada: :+1: First off, thanks for taking the time to contribute! :tada: :+1:

The following is a set of guidelines for contributing to the Script Language Containers. 
These are mostly guidelines, not rules. Use your best judgment, and feel free 
to propose changes to this document in a pull request. 
Furthermore, they contain some information which might help you during development.

## How can you contribute?

You can contribute to this project at several levels:
- Probably, the easiest way to contribute are bug reports or feature requests in the Github Issues
- If, you are more experienced, we are happy about any pull request for bug fixes, improvements or new flavors. 
    - **However, be aware that the Script Language Container are tightly integrated into the Exasol Database, 
    we need to check any contribution thoroughly and might reject pull requests which may break this integration. 
    So, please open first a Github Issue and discuss the changes with us.**
        - The implementation of the script client is especially such a area, 
        which might break many things, if you are not careful
    - **And please respect, that we do not except changes that add packages to the build step 
    `flavor-customization` of any flavor, because this step should be used by users for 
    temporary additions to the flavor. If you would like to add new dependencies to a flavor, 
    please add them to the corresponding build step:**
        - `udfclient-deps:` dependencies which are required for the script client to run, 
        which need to be in the final container
        - `language-deps:` dependencies for the script language of the flavor
        - `build-deps:` any dependencies which are required to build the script client,
        but which are not needed in the final container
        - `flavor-base-deps(_2):` dependencies which are flavor specific
        - `base-test-deps:` dependencies which are only needed for development, debugging and testing
    
        If you are not sure, where to add the dependencies, 
        do not hesitate to open an Github Issue to discuss the change with us.
        
## Testing

We use tests on different levels, we have unit and integration tests for exaslct (Script Language Container Tool, 
our build system for the containers) and we have integration tests for all flavors with the Exasol 
[docker-db](https://github.com/exasol/docker-db). 
You can see in our travis configuration how we run the tests.

The tests for exaslct are located [here](exaslct_src/test). 
They consists of several python unittest tests and 
defines a test flavor in the [resources](exaslct_src/test/resources)
which is used for integration tests.

The integration tests for the flavors are located [here](tests). 
They consists of generic language tests and flavor specific tests.
You can execute them with the exaslct run-db-test command.

## How to configure Travis

If you want to use Travis as your continuous integration server during development 
you need to configure it in a very specific way. Our build takes a while, 
because we build quite extensive containers for some flavors. 
Furthermore, our integrations tests take their time, too. 
Therefore, we were forced to use caching between the build stages of Travis. 
Unfortunately, the existing Travis Cache does not handle caching of artifacts 
from multiple jobs in a stage very well, such that we had to use a 
external cloud service for caching.

Because a build of a flavor already consists of a sequence of docker images, 
we decided to use docker registries as build cache. You can use either docker hub or 
your own docker registry as cache. We encode the information about the docker registry
as encrypted environment variables in the .travis.yml. If you want to use Travis in 
your fork of this repository you have to set your own encrypted environment variables.
Please, revert the environment variables before you create pull request to the original ones, 
used in [exasol/script-languages](https://github.com/exasol/script-languages) repository, 
such that we can test your changes before the merge.

We use the following encrypted Environment Variables 
to provide the Information for the docker registry to the build:

- DOCKER_REPOSITORY
- DOCKER_USERNAME
- DOCKER_PASSWORD

The DOCKER_REPOSITORY needs to be of the following form 

    <hostname>[:port]/<user>/<repository-name>
    
For more information, about encrypted Environment Variables in travis, 
please check the [travis documentation](https://docs.travis-ci.com/user/environment-variables/#defining-encrypted-variables-in-travisyml)