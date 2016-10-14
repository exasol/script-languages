# Python2 and Python3 Client

## Overview

This script language client that implements Python2 and Python3 as a
language for EXASOL. It is implemented in pure Python and uses the
same source code for both languages.  Users can provide Python code in
CREATE SCRIPT statements. The code is compiled on the fly and then
executed.  The client is organized like this: The main folder contains
`common_client.py` which implements the actual language client.  The
subfolders `python2` and `python3` include small wrapper scripts which
invoke the appropriate Python interpreter and then import the common
code.

## Building the client

The client depends on the existence of Python version of the Protobuf
code in `../script_client.proto`, which will be downloaded and
archived together with the actual client.

We recommend to build the client in the same Linux container that it
 is later run in.  For this purpose, we can either build one ourselfs
 from scratch like described in the documentation about the EXASOL
 Linux container for script languages, or we simply import the one the
 is installed in your version of EXASOL.  For instance, if you have
 EXASOL in a virtual machine with a local ip addree 192.168.56.104 on
 your development computer and have configured default BucketFS to
 listen to port 2580 via EXAoperation, the pre-installed Linux
 container can be imported as follows:


```
docker import http://192.168.56.104:2580/default/EXAClusterOS/ScriptLanguages-6.0.0.tar.gz mydockname
```

Now we start the container and share the folder `py` like this:

```
docker run -v `pwd`/python_client:/py --name=mydockname -it mydockname /bin/bash
```

Now we can build the client by typing `make` in the `/py` folder:

```
$ docker run -v `pwd`/python_client:/py --name=sl02 -it sl02 /bin/bash
root@d76fab66fd20:/# cd py/
root@d76fab66fd20:/py# make
wget https://raw.githubusercontent.com/EXASOL/script-languages/master/script_client.proto
--2016-10-14 20:36:13--  https://raw.githubusercontent.com/EXASOL/script-languages/master/script_client.proto
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.60.133
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.60.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 8011 (7.8K) [text/plain]
Saving to: 'script_client.proto'

script_client.proto              100%[=========================================================>]   7.82K  --.-KB/s    in 0s      

2016-10-14 20:36:13 (48.3 MB/s) - 'script_client.proto' saved [8011/8011]

protoc -I. script_client.proto  --python_out=.
tar -zcf pythonclient.tar.gz common_client.py python2 python3 script_client_pb2.py
```


## Deploying the client

After building the client, exit docker and upload the client into BucketFS. Here we assume that you have created a bucket named `py` (with write-password `writepw`) in the default BucketFS:

```
curl -vX PUT -T pythonclient.tar.gz http://w:writepw@192.168.56.104:2580/py/pythonclient.tar.gz
```

Finally, in order to use our new version of Python2 and Python3 in SQL, you need to inform the SQL compiler about the new languages. To do so in your current SQL session, you need to modify the session/system parameter SCRIPT_LANGUAGES, for instance like this:

```
alter session set script_languages = 'PYTHON=builtin_python R=builtin_r JAVA=builtin_java PY2=localzmq+protobuf:///bfsdefault/default/EXAClusterOS/ScriptLanguages-6.0.0#buckets/bfsdefault/py/pythonclient/python2/client PY3=localzmq+protobuf:///bfsdefault/default/EXAClusterOS/ScriptLanguages-6.0.0#buckets/bfsdefault/py/pythonclient/python3/client';
```


Note:If you are using `alter session`, you need to re-issue the command above when you start a new session.

## Example
Now Python2 in pure Python and Python3 is available as script language:

```
create or replace PY2 scalar script p2() returns double as
from math import pi
def run(c):
    return pi * 2
/

create or replace PY3 scalar script p3() returns double as
import math
def run(c):
    return math.pi * 3
/

```