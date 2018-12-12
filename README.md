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

For running the tests you also need
* Python
* GCC
* Java
* unixODBC
* Docker with privileged mode

## Quickstart
1. Choose a flavor. Currently we have several pre-defined flavors available, e.g., `mini`, `standard` and `conda`.
This project supports different versions of script language environments with different libraries and languages.
We call these versions _flavors_. The pre-defined flavors can be modified and extended to create customized flavors.
Each pre-defined flavor has its own set of Docker build-files in a corresponding subfolder of [flavors](flavors).
2. Create the language container. We choose to use the `mini` flavor which is the smallest of the currently available flavors and which only support the Python language.
```bash
$ ./build --flavor=mini
```
(on Mac OS X, use `./build -f mini`)

3. Export it into a standalone archive
```bash
$ ./export --flavor=mini
```
(on Mac OS X, use `./export -f mini`)
This creates the file `mini.tar.gz`.

Optionally, you can run some automated tests for your flavor by using
```bash
$ ./test_complete --flavor=mini
```
(on Mac OS X you need to have pip installed for this to work)
If the test fails with the message
```
cp: cannot create regular file ‘/tmp/udftestdb/exa/etc/EXAConf’: Permission denied
```
then run `stop_dockerdb` and restart the test.

4. Upload the file into bucketfs. For the following example we assume the password `pwd` and the bucketname `funwithudfs` in a bucketfs that is running on port `2580` on machine `192.168.122.158`
```bash
curl -v -X PUT -T mini.tar.gz w:pwd@192.168.122.158:2580/funwithudfs/mini.tar.gz
```
5. In SQL you activate the Python implementation of the flavor `mini` by using a statement like this
```sql
ALTER SESSION SET SCRIPT_LANGUAGES='MYPYTHON=localzmq+protobuf:///bucketfsname/funwithudfs/mini?lang=python#buckets/bucketfsname/funwithudfs/mini/exaudf/exaudfclient';
```
Now the script language `MYPYTHON` can be used to define a script, e.g., 
```
CREATE SCHEMA S;
CREATE mypython SCALAR SCRIPT small_test() RETURNS DOUBLE AS
def run(ctx):
   return 1.2
/
```
The script can be executed as follows:
```
select small_test();
```
6. Afterwards you may choose to remove the Docker images for this flavor. This can be done as follows:
```bash
./clean --flavor=mini
```
Please note that this script does not delete the Linux image that is used as basis for the images that were build in the previous steps.



