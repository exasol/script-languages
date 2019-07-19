#!/bin/bash
set -o errexit
set -o pipefail
set -x
JAVA_PREFIX=$1
CLIENT_BINARY=$2
JAVA_RPATH=$(find "$JAVA_PREFIX/lib" -name '*.so' | xargs -n 1 dirname | sort | uniq | grep -v jli | paste -sd ":" -)
echo "New Java rpath: $JAVA_RPATH"
RPATH_WITHOUT_JAVA=$(chrpath -l "$CLIENT_BINARY" | sed "s#$CLIENT_BINARY: RUNPATH=##g" | tr ':' '\n' | xargs -n 1 | grep -v java | paste -sd ":" -)
echo "Original rpath without Java: $RPATH_WITHOUT_JAVA"
RPATH="$JAVA_RPATH:$RPATH_WITHOUT_JAVA" #:/usr/lib/x86_64-linux-gnu/jni/
chmod u+w "$CLIENT_BINARY"
chrpath -r "$RPATH" "$CLIENT_BINARY"
chmod u-w "$CLIENT_BINARY"
