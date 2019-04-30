# EXASLCT (Script Langauge Container Tool)
EXASLCT is the build tool for the script language container. 
Its usage is described [here](../README.md). 
This readme is about the inner working of EXASLCT.

## About the script language containers

The script language containers are getting build from several Dockerfiles which depend on each other. 
These Dockerfiles need to install all necessary dependencies for the [script client](../src), 
compile the script client and install all necessary dependencies for the flavor and the customizations of the user.

## Problem Statement:
The old-style to build the containers was slow and tidious. 
A single change in the dependencies required a rebuild of everything. 
Furthermore, essential dependencies for the script client were 
mixed with flavor dependent dependencies. 
Changes were difficult for users and could break the container. 
It was acutal unclear which of the dependencies were essential which were not.
For some flavors it was impossible to run the build on travis,
because it exceeded the maximum runtime per job of 50 minutes. 
The build system and the test runner were bash script 
which were messy and difficult to maintain. 
They worked with background jobs to do thing in parallel 
which convoluted the logs and error analysis 
if things went wrong were difficult. 
Further, the test runner left temporary files which were owned by root, 
because they were created by a docker container.

## Design Goals:

* Easy customization of existing flavors for the user
* Separation of concern for code and Dockerfiles
* Parallel builds
* Partial builds
* Local and remote caching
* Faster development cycles for the script client
* Allowing output redirection for testing of the flavor
* Encapsulate running the tests and all its dependencies 
  into docker containers or volumes.
  
## Programming Model
  
## How does it work?

* Exaslct is mix of a build system and test runner with infrastructure as code
* The basic design pattern comes here from the build system which defines tasks 
  which can have dependencies on other tasks, execute something and 
  produce a output.
* With this design pattern we can describe the dependencies of the Dockerfiles 
  with Task which somehow create a Docker Image and  return this to dependent tasks.
  These might create another Docker Image and return this to other Tasks. 
  We get a direct acyclic graph of dependent tasks.
* Also the test runner can be formulated as a direct acyclic graph of dependencies.
  For example, we need to spwan a test environment with docker-db, 
  upload and build the script language container and load data before we can execute tests.
* We use [Luigi](https://luigi.readthedocs.io/en/stable/) to describe and 
  execute the tasks and their dependencies. Luigi was invented to describe data science workflow 
  as tasks and their dependencies, but is also suitable for other workflows