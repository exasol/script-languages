# EXASOL Script Languages
[![Build Status](https://travis-ci.org/EXASOL/script-languages.svg?branch=port-exaudfclient)](https://travis-ci.org/EXASOL/script-languages)

###### Please note that this is an open source project which is officially supported by EXASOL. For any question, you can contact our support team.

## Table of Contents
1. [About](#about)
2. [Prerequisites](#prerequisites)
3. [Quickstart](#quickstart)

<!--
4. [Overview](#overview)
5. [EXASOL script language protocol](#exasol-script-language-protocol)
6. [Linux Container for script languages](#linux-container-for-script-languages)
7. [Script language clients](#script-language-clients)
-->

## About
This project contains implementations for user defined functions that can be used in the EXASOL database (version 6.0.0 or later)

## Prerequisites
In order to build this project, you need
* Linux or MacOS X (not really test yet)
* Docker

In order to follow the quickstart guide, you additionally need
* Write-access to a bucket in a bucketfs in an EXASOL installation
* curl
* A SQL client connecting to the same EXASOL installation

## Quickstart
0. Choose a flavor. Currently we have the following available: `mini`, `standard`, `conda`.
This project support different versions of script language environments with different sets of libraries and languages.
We call these versions _flavors_ for no particular reason.
Each flavor has its own set of build-files in a corresponding subfolder of [dockerfiles](dockerfiles).
1. create the language container (we choose to use the `mini` flavor which is the smallest of the currently available flavors and only support the Python language)
```bash
$ ./build --flavor=mini
```
2. export it into a standalone archive
```bash
$ ./export --flavor=mini --target=myminiudfs
```
This creates the file `myminiudfs.tar.gz`
3. Upload the file into bucketfs (we assume the password `w` and the bucketname `funwithudfs` in a bucketfs that is running on port `2580` on machine `192.168.122.158`)
```bash
curl -v -X PUT -T myminiudfs.tar.gz w:w@192.168.122.158:2580/funwithudfs/myminiudfs.tar.gz
```
4. In SQL you activate the language implementation by using a statement like this
```sql
ALTER SESSION SET SCRIPT_LANGUAGES='MYPYTHON=localzmq+protobuf:///bucketfsname/funwithudfs/myminiudfs?lang=python#buckets/bucketfsname/funwithudfs/myminiudfs/exaudf/exaudfclient';
```
5. Afterwards you may choose to remove the docker images for this flavor
```bash
./clean --flavor=mini
```

(Please note, at the moment we do not delete the Linux image that is used as basis)


<!--
## Overview

EXASOL is shipped with the support for various script languages: Java, Python, R and Lua. But 
EXASOL's script framework allows to expand its language support in two dimensions. You can upload
additional libraries and packages for the existing languages. But if you want to run other programming
languages directly within the EXASOL cluster, you can even install these languages by your own.

Scripts can be used in EXASOL in two main areas. On the one hand, user-defined functions (UDFs) 
can be used to run all kinds of scalar, aggregate and analytical logic on table data. You can even 
create MapReduce scripts which are directly integrated in normal SQL queries. On the other hand,
the so-called adapter scripts implement the underlying logic for virtual schemas. Virtual schemas 
contain virtual tables which look and behave like normal, persistent tables in EXASOL. If you query 
that data, the adapter script logic cares about how to transfer the actual data from the specified
external data source, and what parts of the SQL query can be pushed down to that underlying system.

Before you start to integrate new languages, we highly recommend to read section `UDF scripts` and 
in particular the subsection `Expanding script languages using BucketFS` in the EXASOL user manual.
This will give you a good overview about the general concepts and the following technical topics:

* Basics of EXASOL UDFs
* How to use BucketFS, EXASOL's synchronous cluster files system
* What script clients are and how they can be built
* How to define a script language alias in EXASOL in your session/system

In this project, we provide a detailed documentation for the communication API between EXASOL and 
script clients and implementations of these clients that you can upload and use on your system. 
If you want to integrate further languages, we would be very happy if you give us your feedback or 
even start contributing to that open source project so that others in the community can take advantage 
of your development.


## EXASOL script language protocol

When scripts in EXASOL are used, the database starts virtual machines in a safe environment using a 
Linux container and the corresponding script client. 

Note: Lua-scripts are different because they are compiled directly into the database engine.

The communication protocol between the virtual machines and the database are based on two main technologies:
* ZeroMQ library (http://zeromq.org/) for createing ipc sockets
* Google's Protocol Buffers (https://github.com/google/protobuf) for encoding messages

The protocol details can be found in file [script_client.proto](script_client.proto)

In general, the following message types have to be implemented by the script client:

* __MT_CLIENT__:  The script language implementation is alive and requests more information
* __MT_INFO__: Basic information about the EXASOL system and cluster configuration and the UDF script code
* __MT_META__:  Names and data types of the data to send between EXASOL and the script language implementation
* __MT_CLOSE__: Terminates the connection to EXASOL
* __MT_IMPORT__: Request the source code of other scripts or information stored in CONNECTION objects
* __MT_NEXT__: Request more data to be sent fom EXASOL to the UDF
* __MT_RESET__: Restart the input data iterator to the beginning
* __MT_EMIT__: Send results from the UDF to EXASOL
* __MT_RUN__: Change status to indicate the start of data transfers
* __MT_DONE__: Indicate that the UDF will send no more results for the current group of data
* __MT_CLEANUP__: Send to indicate that no more groups of data will have to be processed by the script language implementation and that it may stop
* __MT_FINISHED__: Sent when the script language implementation successfully stopped
* __MT_CALL__: Used to call a certain function in the UDF when in Single-Call mode
* __MT_RETURN__: Used to send the result of the Single-Call function call
* __MT_UNDEFINED_CALL__: Sent when a script does not implement a requested single-call function

There exist two *deprecated* message types which were used for eUDFs:
_MT_PING_PONG_, _MT_TRY_AGAIN_


Here is a sketch of the interaction between EXASOL and script language
client when using the language in a UDF context:

After being started by the database, the language implementation takes
initiative, it sends a message of type MT_CLIENT to the database,
which responds with a message of type MT_INFO which among general
information about the EXASOL system and the UDF (like database name,
database, the number of cluster nodes, etc.) also contains the actual
source code of the UDF.  Then the script language implementation
requests the meta data of the current UDF: types and names of input
and output columns, the call mode and the required input and output
iteration behaviors.
 
Finally, the script language implementation notifies the database
system that it is up and running and will be requesting and sending
actual data using the MT_RUN message.

Then the UDF and EXASOL iteratively interchange data. Here, the UDF
requests new data with MT_NEXT messages, and sends results with
MT_EMIT messages.

A different context of use is single-call mode, here, the script is
started and a single function is called in the script. Single-call
mode is used when the SQL Compiler uses information that is provided
via scripts, for instance when virtual schemas are defined using
adapter scripts.

The details of the protocol can best be understood by examining the
source code of the example implementations.
 

## Linux Container for script languages

In order to create the Linux container which is used to abstract the
script language implementation from EXASOL, we use Docker
(https://www.docker.com) as a _build tool_. Inside the EXASOL cluster
We do __not__ use Docker in order to _run_ script language
implementation.

In addition to the Docker file which describes how to build a first
version of the Linux container, we also provide a small shell script
in folder [linux_container](linux_container) which cares about some 
packaging details.

###### Please note that all files in the Linux container are owned by the same operating system user that is used to run the UDF. This means that files in the Linux container like /etc/shadow are readable by any database user that can execute UDF scripts. As a consequence, when creating a new Linux container, you have to make sure that it does not contain any sensitive data. If you need to access these kind of data in UDFs, please use a __private__ bucket to store them.

## Script language clients

Some examples of script language implementations are provided in the
corresponding subfolders, including the details about how to build these
clients.
-->
