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
* User customizations are not able to break the container
* Separation of concern for code and Dockerfiles
* Parallel builds
* Partial builds
* Local and remote caching
* Faster development cycles for the script client
* Allowing output redirection for testing of the flavor
* Encapsulate running the tests and all its dependencies 
  into docker containers or volumes.
  
## Programming Model

Exaslct is a mix of a build system, test runner and infrastructure as code. 
As such, we typically have tasks like the following one:
    * Build Image
    * Start Container
    * Upload something
Most of these tasks produce some kind of output, for example:
    * docker image
    * a running docker container
Often, other tasks than depend either on the output or the action of one or more other tasks.
Together, this results in a direct acyclic graph of tasks, also known as workflow. 
Tasks that depend on each other will run in sequence, 
but that are independent of each other may run in parallel. 
This also allows a good separation of concern.
As workflow executor we use  [Luigi](https://luigi.readthedocs.io/en/stable/) 
which was actually developed for batch data science workflows, 
but is suitable for different scenarios, too.
Luigi describes tasks as subclasses of Luigi.Task which implements the following methods:

```python
class TaskC(luigi.Task):

    def output(self):
        return Target()
    
    def run(self):
        #do somthing
        pass
        
    def requires_tasks(self):
        return [TaskA(),TaskB()]

```

Here we describe a TaskC which depends of TaskA and TaskB defined in the requires() method. 
It does something which is specified in the run() method. 
Futher, it produces Target() as output. 
Luigi provides the dependency resolution, scheduling and parallelisation.

Besides this static way of describing the dependencies between tasks, 
Luigi also provides so called [dynamic dependencies](https://luigi.readthedocs.io/en/stable/tasks.html#dynamic-dependencies), 
which allow more flexible patterns in special case. 
Especially, if the order of execution of dependencies is important or 
the dependencies depend on some calculation. The dynamic dependencies 
allow the implementation of a fork-join pattern.

## How are the dependencies between the build stages

We compose the langauge container from several different Dockerfiles.
Each Dockerfile install dependencies for one specific purpose.
We also added a separate Dockerfile flavor-customization for user specific changes.
The user specifc changes will be merged on filesystem basis 
with the resulting docker images for the udfclient. 
The merge will overwrite user specific changes 
that could prevent the udfclient from working properly.

![](docs/image-dependencies.png)

A dependency between build stages can be either a FROM or 
COPY dependencies. A FROM dependency means that 
the target of the arrow uses the source of the arrow as base image.
A COPY dependency means that the target of the arrow [copies parts](https://docs.docker.com/develop/develop-images/multistage-build/) of 
the source of the arrow.

All stages with the string "build_run" in their name, 
either run the build for the udfclient or 
at least inherit from a images which had build it. 
As such these images contain all necassary tools to rebuild 
the udfclient for debugging purposes.

## How does caching work