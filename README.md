# EXASOL Script Languages
[![Build Status](https://travis-ci.org/exasol/script-languages.svg?branch=master)](https://travis-ci.org/exasol/script-languages)

###### Please note that this is an open source project which is officially supported by EXASOL. For any question, you can contact our support team.

## Table of Contents
1. [About](#about)
2. [Prerequisites](#prerequisites)
3. [Quickstart](#quickstart)

## About
This project contains implementations for user defined functions (UDF's) that can be used in the EXASOL database (version 6.0.0 or later)

## Prerequisites
In order to build this project, you need:
* Linux
* Docker

In order to follow the quickstart guide, you additionally need
* Write-access to a bucket in a bucketfs in an EXASOL installation
* curl
* An SQL client connecting to the same EXASOL installation

For running the tests you also need Python, GCC, Java and unixODBC.

## Quickstart
1. Choose a flavor. Currently we have several pre-defined flavors available, e.g., `mini`, `standard` and `conda`.
This project supports different versions of script language environments with different libraries and languages.
We call these versions _flavors_. The pre-defined flavors can be modified and extended to create customized flavors.
Each pre-defined flavor has its own set of Docker build-files in a corresponding subfolder of [flavors](flavors).
2. create the language container (we choose to use the `mini` flavor which is the smallest of the currently available flavors and only support the Python language)
```bash
$ ./build --flavor=mini
```
3. export it into a standalone archive
```bash
$ ./export --flavor=mini
```
This creates the file `mini.tar.gz`.

You can optionally run some automated tests for your flavor by using
```bash
$ ./test --flavor=mini
```
4. Upload the file into bucketfs (we assume the password `w` and the bucketname `funwithudfs` in a bucketfs that is running on port `2580` on machine `192.168.122.158`)
```bash
curl -v -X PUT -T mini.tar.gz w:w@192.168.122.158:2580/funwithudfs/mini.tar.gz
```
5. In SQL you activate the language implementation by using a statement like this
```sql
ALTER SESSION SET SCRIPT_LANGUAGES='MYPYTHON=localzmq+protobuf:///bucketfsname/funwithudfs/mini?lang=python#buckets/bucketfsname/funwithudfs/mini/exaudf/exaudfclient';
```
6. Afterwards you may choose to remove the docker images for this flavor from the local Docker repository
```bash
./clean --flavor=mini
```

Please note, that this script does not delete the Linux image that is used as basis from the local Docker repository.



