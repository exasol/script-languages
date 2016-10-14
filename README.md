# EXASOL Script Languages

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Overview](#overview)
3. [EXASOL script language protocol](#exasol-script-language-protocol)
4. [Linux Container for script languages](#linux-container-for-script-languages)
5. [Script language clients](#script-language-clients)


## Prerequisites
The following topics are not part of this documentation:
* Basics of EXASOL user defined functions
  * Iteration types: SET-EMITS
* How to use BucketFS
  * The HTTP perspective
  * The UDF perspective
* How to define a script language alias in EXASOL
* How to use an alternative Linux container with the languages that are come pre-installed with EXASOL

Before starting out, please read section `UDF scripts` and in particular the subsection `Expanding script languages using BucketFS` in the EXASOL user manual.

## Overview
In EXASOL, user defined function scripts (UDF) usually run in parallel and
distributed in a cluster. This means, that for a UDF in a given
language, there are typically many instances of the script language
implementation running at the same time.

Usually each of these instances only processes a small subset of the
data (the only exception being are UDFs with input-output behavior
SET-RETURNS and SET-EMITS)

Another context of using scripts is when they provide call-backs for the SQL compiler (like for instance, when used as an _adapter script_ for _virtual schemas_).

For all script languages (except for Lua which is directly compiled
into the SQL Engine) EXASOL starts the instances of the language
implementation in dedicated Linux containers in order to protect the
datbase from misbehaving scripts and language implementations.
Script language implementation and EXASOL then communicate via the protocol which is outlined [below](#exasol-script-language-protocol).

When creating new script languages for EXASOL, users therefore need to do the following:
* provide a Linux container (or use the same as EXASOL for its pre-installed languages)
* create the actual implementation of the script language, the __script client__


## EXASOL script language protocol

When EXASOL a script language client, the first argument always is the name of a local ipc socket which is created by EXASOL using the ZeroMQ library (http://zeromq.org/).

The messages between EXASOL and an UDF script language
implementation are encoded using Google's Protocol
Buffers (https://github.com/google/protobuf).
The following message types are used:

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

The following message types are defined in the `.proto` file but are only used to communicate with eUDFs (which are not described here): _MT_PING_PONG_, _MT_TRY_AGAIN_


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

In order to create the Linux container which used to isolate the
script language implementation from EXASOL, we use Docker
(https://www.docker.com) as a _build tool_. Inside the EXASOL cluster
We do __not__ use Docker in order to _run_ script language
implementation.

In addition to the Docker file which describes how to build a first
version of the Linux container, we also provide a small shell script
that adds some final touches.

## Script language clients

Some examples of script language implementations are provided. The
details on how to build these clients can be found in the respective
subdirectory.

* __cpp_client__: a script language client that implements C++ as a
  language for EXASOL. Users can provide C++ code in CREATE SCRIPT
  statements. The code is compiled on the fly and then executed.

* __python_client__: an pure Python implementation of Python 2 and Python 3 as
  language for EXASOL.