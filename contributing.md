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


The tests for exaslct are located [here](exaslct_src/test). 
They consists of several python unittest tests and 
defines a test flavor in the [resources](exaslct_src/test/resources)
which is used for integration tests.

The integration tests for the flavors are located [here](tests). 
They consists of generic language tests and flavor specific tests.
You can execute them with the exaslct run-db-test command.
