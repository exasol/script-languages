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

## Basic Example
Now C++ is available as script language:

```
create or replace cpp scalar script csin(x double) returns double as

#include <cmath>

void run_cpp(SWIGMetadata& meta, SWIGTableIterator& iter, SWIGResultHandler& res)
{
        res.setDouble(0,sin(iter.getDouble(0)));
        res.next();
}

/

select csin(32);
```


### Example of an aggregating UDF (set-returns)
This example shows how to iterate over a incoming set of data by using
the `next()` method of the `SWIGTableIterator`.
Please note that C++ UDFs don't make a difference between "returning" and "emitting" values.
Both behaviors are achieved by setting a result value and then calling `next()` on the `SWIGResultHandler`.
For UDFs that return a value, use the column 0 to put the result value.

```
create or replace cpp set script sum_of_squares_cpp(x float) returns float as

void run_cpp(SWIGMetadata& meta, SWIGTableIterator& iter, SWIGResultHandler& res)
{
    double acc = 0.0;
	do {		
		double current = iter.getDouble(0);
        acc += current*current;
    } while (iter.next());
    res.setDouble(0,acc);
    res.next();
}
/
```

Let's experiment a litte:

```
create or replace python scalar script fill(n int) emits (x float) as
def run(ctx):
    for i in range(ctx.n):
        ctx.emit(i)
/

create or replace table vals as select fill(1000000);

select sum_of_squares_cpp(x) from vals;
```
On my laptop, this took 2.3 seconds.

Here is an alternative version as Python UDF:

```
create or replace python set script sum_of_squares_py(x float) returns float as
def run(ctx):
    acc = 0.0
    while True:
        acc += ctx.x*ctx.x
        if not ctx.next(): break
    return acc
/

select sum_of_squares_py(x) from vals;
```
On my laptop, this took 5.6 seconds.

Please note, that while the C++ performance is nice for a UDF, it is nowhere near what pure SQL can do on EXASOL:
```
select sum(x*x) from vals;
```
takes 0.1 seconds on my machine.


### Example of an emitting UDF
This example shows how to output multiple values for EMITS scripts.
Basically, this is achieved by calling the `next()` method of the `SWIGResultHandler` multiple times.

```
create or replace cpp set script duplicate_rows_cpp(x int) emits (x int) as

void run_cpp(SWIGMetadata& meta, SWIGTableIterator& iter, SWIGResultHandler& res)
{
	do {
        int64_t current = iter.getInt64(0);
        res.setInt64(0,current);
        res.next();  // first emit per input row
        res.setInt64(0,current);
        res.next();  // second emit per input row
    } while (iter.next());
}
/


select duplicate_rows_cpp(x) from vals;
```

On my laptop, this took 2.7 seconds.

A Python version could look like this:
```
create or replace python set script duplicate_rows_py(x int) emits (x int) as
def run(ctx):
    while True:
        ctx.emit(ctx.x)
        ctx.emit(ctx.x)
        if not ctx.next(): break
/


select duplicate_rows_py(x) from vals;
```

For me, this took 11.9 seconds.

Here is a Lua version:
```
create or replace lua set script duplicate_rows_lua(x int) emits (x int) as
run = function(ctx)
    while true do
        ctx.emit(ctx.x)
        ctx.emit(ctx.x)
         if not ctx.next() then break end
    end
end
/

select duplicate_rows_lua(x) from vals;
```

Please note, that normally Lua scripts tend to be much faster in EXASOL than scripts in other languages, because Lua is tighter integrated with EXASOL.
Therefore it comes as quite a surprise that the Lua version, taking 2.3 seconds on my machine is only slightly faster than the C++ version. Even more so when considering that the current version of the C++ UDF language implementation compiles the C++ code on the fly which has a really large overhead.
Check this:
```
select duplicate_rows_cpp(1);
```
while

```
select duplicate_rows_py(1);
```
only takes 0.1 seconds (as does the Lua version)

So clearly, the overhead of compiling C++ is quite large at the moment!