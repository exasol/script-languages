# CPP Client

## Overview
This script language client implements C++ as a
language for EXASOL. Users can provide C++ code in CREATE SCRIPT
statements. The code is compiled on the fly and then executed.

## Building the client

In order to build the client example code, the
best strategy is to locally start the very same Linux container that
is later used to run the client in in the EXASOL database.

For this purpose, we suggest using Docker (https://www.docker.com).

* First we need to get the Linux on the development machine.  For this
purpose, we can either build one ourselfs from scratch like described
in the documentation about the EXASOL Linux container for script
languages, or we simply import the one the is installed in your version of EXASOL.
For instance, if you have EXASOL in a virtual machine with a local ip addree 192.168.56.104 on your development computer and have configured default BucketFS to listen to port 2580 via EXAoperation, the pre-installed Linux container can be imported as follows:


```
docker import http://192.168.56.104:2580/default/EXAClusterOS/ScriptLanguages-2017-01-30.tar.gz mydockname
```
Now we start the container and share the folder `src` like this:

```
docker run -v `pwd`/src:/src --name=mydockname -it mydockname /bin/bash
```

`src` is no mounted as `/src` inside the container:

```
root@d673f112aaca:~# cd /src
root@d673f112aaca:/src# ls
Makefile  cpp.h         scriptDTO.h          scriptDTOWrapper.h  zmqcontainer.proto
cpp.cpp   scriptDTO.cc  scriptDTOWrapper.cc  swigcontainers.h    zmqcontainerclient.cc
```

Typing `make` downloads some dependencies, builds the client and stores it in `cppclient.tar.gz`.
Note: there may also be some warning regarding the use of a deprecated feature in the jsoncpp library.

## Deploying the client

After building the client, exit docker and upload the client into BucketFS. Here we assume that you have created a bucket named `cpp` (with write-password `writepw`) in the default BucketFS:

```
curl -vX PUT -T src/cppclient.tar.gz http://w:writepw@192.168.56.104:2580/cpp/cppclient.tar.gz
```

Finally, in order to use C++ in SQL, you need to inform the SQL compiler about the new language. To do so in your current SQL session, you need to modify the session/system parameter SCRIPT_LANGUAGES, for instance like this:

```
alter session set script_languages = 'PYTHON=builtin_python R=builtin_r JAVA=builtin_java CPP=localzmq+protobuf:///bfsdefault/default/EXAClusterOS/ScriptLanguages-2017-01-30?lang=cpp#buckets/bfsdefault/cpp/cppclient/cppclient';
```

Note: as we are using `alter session`, you need to re-issue the command above when you start a new session.
An alternative would be to use `alter system`.

## Example
Now C++ is available as script language:

```
create or replace cpp scalar script csin(x double) returns double as

#include <cmath>

void run_cpp(const SWIGMetadata& meta, SWIGTableIterator& iter, SWIGResultHandler& res)
{
        res.setDouble(0,sin(iter.getDouble(0)));
        res.next();
        res.flush();
}

/

select csin(32);
```
