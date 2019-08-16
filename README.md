# EXASOL Script Languages
[![Build Status](https://travis-ci.org/exasol/script-languages.svg?branch=master)](https://travis-ci.org/exasol/script-languages)

###### Please note that this is an open source project which is officially supported by EXASOL. For any question, you can contact our support team.

## Table of Contents
1. [About](#about)
2. [Prerequisites](#prerequisites)
3. [How to build an existing flavor?](#how-to-build-an-existing-flavor)
4. [How to customize an existing flavor?](#how-to-customize-an-existing-flavor)
5. [Partial builds or rebuilds](#partial-builds-and-rebuilds)
6. [Using your own remote cache](#using-your-own-remote-cache)
7. [Testing an existing flavor](#testing-an-existing-flavor)
8. [Cleaning up after your are finished](#cleaning-up-after-your-are-finished)

## About
This project contains script language containers for user defined functions (UDF's) 
that can be used in the EXASOL database (version 6.0.0 or later). 
A script language container consists of a Linux container with a complete linux distribution and all required libraries, 
such as a script client. A script client is responsible for the communication with the database and for executing the script code.
We provide in this repository several [flavors](flavors) of script language containers, 
such as the current standard implementation of the [script client](src) with support for Python 2/3, R und Java. 
We will show here how to customize and build the different flavors of the script language containers. 
Pre-built containers can you find in the [release section](https://github.com/exasol/script-languages/releases) of this repository.
If you are interested in the script client you find more details [here](src/README.md).

## Prerequisites
In order to build this project, you need:
* Linux or Mac OS X (experimental)
* Docker >= 17.05 [multi-stage builds required](https://docs.docker.com/develop/develop-images/multistage-build/)
* Python >=3.6 with pip
* We recommend at least 50 GB free disk space on the partition 
  where Docker stores its images, on linux Docker typically stores 
  the images at /var/lib/docker.
* For the partition where the output directory (default: ./.build_output) 
  is located we recommend additionally at least 10 GB free disk space.

Further, prerequisites might be necessary for specific tasks. These are listed under the corresponding section.

## How to build an existing flavor?

Choose a flavor. Currently we have several pre-defined flavors available, e.g., `mini-EXASOL-6.0.0`, and `standard-EXASOL-6.1.0`.
This project supports different versions of script language environments with different libraries and languages.
We call these versions _flavors_. The pre-defined flavors can be modified and extended to create customized flavors.
Each pre-defined flavor has its own set of Dockerfiles in a corresponding subfolder of [flavors](flavors).

Create the language container and export it to the local file system

```bash
$ ./exaslct export --flavor-path=flavors/<flavor-name> --export-path <export-path>
```

or upload it directly into your BuckerFS (currently http only, https follows soon)

```bash
$ ./exaslct upload --flavor-path=flavors/<flavor-name> --database-host <hostname-or-ip> --bucketfs-port <port> \ 
                   --bucketfs-username w --bucketfs-password <password>  --bucketfs-name <bucketfs-name> \
                   --bucket-name <bucket-name> --path-in-bucket <path/in/bucket>
```

Once it is successfully uploaded, it will print the ALTER SESSION statement
that can be used to activate the script language container in the database.

## How to customize an existing flavor?

To customize an existing flavor you can add your specific needs to the Dockerfile in the flavor-customization directory. 
You can run commands with:

```Dockerfile
RUN <command>
```

For example, to install new software you can use:

```Dockerfile
RUN apt-get -y update && \
    apt-get install \<packages> && \
    apt-get -y clean && \
    apt-get -y autoremove
```
    
You need to run apt-get update, because any previous step clears the cache of apt to keep the docker images small. 
The commands 

```Dockerfile
apt-get -y clean and apt-get -y autoremove 
```

clear the cache.

You can add to the flavor-customization directory additional files which you can use in the Dockerfile via:

```Dockerfile
COPY flavor-customization/<your-file-or-directory> <destination>
```

or 

```Dockerfile
ADD flavor-customization/<your-file-or-directory> <destination>
```

Your changes on the file system will then be merged with the file system of the script client
which contains all necessary libraries that are required to run the script language runtime.

Be aware that the merge will override all changes which may prevent the execution of the script client.
In details, this means if you change or remove packages or files in flavor-customization
which are necessary for the script client they will be restored in the final container.

After you finished your changes, rebuild with 

```bash
$ ./exaslct export --flavor-path=flavors/<flavor-name>
```

or upload it directly into your BuckerFS (currently http only, https follows soon)

```bash
$ ./exaslct upload --flavor-path=flavors/<flavor-name> --database-host <hostname-or-ip> --bucketfs-port <port> \ 
                   --bucketfs-username w --bucketfs-password <password>  --bucketfs-name <bucketfs-name> \
                   --bucket-name <bucket-name> --path-in-bucket <path/in/bucket>
```

Note: The tool `exaslct` tries to reuse as much as possible of the previous build or tries to pull already exising images from Docker Hub.

## Force a rebuild

Sometimes it is necessary to force a rebuild of a flavor. 
A typical reason is to update the dependencies in order to
fix bugs and security vulnerabilities in the installed dependencies.
To force a rebuild the command line option --force-rebuild can be used 
with basically all commands of ./exaslct, except the clean commands.

## Partial builds and rebuilds

In some circumstances you want to build or rebuild 
only some parts of the flavor. Most likely during development or during CI. 
You can specify for a build upper bounds (also called goals) 
until which the flavor should be build and for rebuilds 
you can define lower bounds from where the rebuild get forced.

You can define upper bounds with the commandline option --goal 
for the ./exaslct commands build and push. 
The build command only rebuilds the docker images, 
but does not export a new container.
All other commands don't support the --goal option, 
because they require specific images to be built,
otherwise they would not proceed.

```bash
./exaslct build --flavor-path=<path-to-flavor> --goal <build-stage>
```

If you want to build several different build-stages at once, you can repeat the --goal option.

The following build-stage are currently available:
* udfclient-deps
* language-deps
* build-deps
* build-run
* base-test-deps
* base-test-build-run
* flavor-test-build-run
* flavor-base-deps
* flavor-customization
* release


With the option --force-rebuild-from, you can specify from where the rebuild should be forced.
All previous build-stages before this will use cached versions where possible.
However, if a single stage is built, it will trigger a build for all following build-stages.
The option --force-rebuild-from only has an effect together with the option --force-rebuild, 
without it is ignored.

```bash
./exaslct build --flavor-path=<path-to-flavor> --force-rebuild --force-rebuild-from <build-stage>
```

Similar, as for the --goal option, you can specify multiple lower bounds 
by repeating the --force-rebuild-from with different build-stages.

## Using your own remote cache

Exaslct caches images locally and remotely. 
For remote caching exaslct can use a docker registry. 
The default registry is configured to Docker Hub. 
With the command line options --repository-name 
you can configure your own docker registry as cache. 
The --repository-name option can be used with all 
./exaslct commands that could trigger a build, 
which include build, export, upload and run-db-test commands.
Furthermore, it can be used with the push command which
uploads the build images to the docker registry.
In this case the --repository-name option specifies 
not only from where to pull cached images during the build,
but also to which cache the built images should be pushed.

You can specify the repository name, as below:

```bash
./exaslct export --flavor-path=<path-to-flavor> --repository-name <hostname>[:port]/<user>/<repository-name>
```

## Testing an existing flavor

To test the script language container you can execute the following command:

```bash
$ ./exaslct run-db-test --flavor-path=flavors/<flavor-name>
```

**Note: you need docker in privileged mode to execute the tests**

## Cleaning up after your are finished

The creation of scripting language container creates or downloads several docker images
which can consume a lot of disk space. Therefore, we recommand to remove the Docker images
of a flavor after working with them.

This can be done as follows:

```bash
./exaslct clean-flavor-images --flavor-path=flavors/<flavor-name>
```

To remove all images of all flavors you can use:

```bash
./exaslct clean-all-images
```

**Please note that this script does not delete the Linux image that is used as basis for the images that were build in the previous steps. 
Furthermore, this command doesn't delete cached files in the output directory. The default path for the output directory is .build-output.**
