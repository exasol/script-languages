Compiling unixODBC on AIX from source
=====================================

Problem 1
---------

The library version for unixODBC 2.2.14 and 2.3.0 is wrong (1 instead
of 2). The EXASolution ODBC driver expects library version 2 and
is therefore unable to use unmodified compiles of unixODBC 2.2.14 or
2.3.0.


Diagnosis
---------

Locate the libraries libodbc.a and libodbcinst.a of your compile with
find. If the output of
	
	ar -t -Xany libodbc.a
	ar -t -Xany libodbcinst.a

lists libodbc.so.2 (or libodbcinst.so.2, resp.) as a member, the library
version is 2.


Solution
--------

As the library version is fixed upstream with unixODBC 2.3.1, the easiest
solution is to avoid unixODBC 2.2.14 and 2.3.0.



Problem 2
---------

Compiling unixODBC from source may fail on AIX 6.1 and AIX 7.1. And even
if compiling works, applications may be unable to find the unixODBC driver
manager or the EXASolution ODBC driver.


Diagnosis
---------

Locate the libraries libodbc.a and libodbcinst.a of your compile with
find. If the output of
	
	ar -t -Xany libodbc.a
	ar -t -Xany libodbcinst.a

lists hundreds of files instead of just a few libraries, recompile as
described below.



Solution
--------

Before compile unixODBC on AIX 6.1 or AIX 7.1, fix the build scripts
in the following way:

	cp config.guess config.guess.old
	sed -e 's/\*:AIX:\*:\[45/&67/' config.guess.old > config.guess
	cp configure configure.old
	sed -e 's/aix5\*/&|aix6*|aix7*/g;' configure.old > configure

This patch is tested for unixODBC 2.2.12 to 2.3.1 with GCC.
