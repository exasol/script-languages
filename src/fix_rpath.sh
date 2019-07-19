#!/bin/bash
set -o errexit
set -o pipefail
INPUT_CLIENT_BINARY=$1
OUTPUT_CLIENT_BINARY=$2
cp "$INPUT_CLIENT_BINARY" "$OUTPUT_CLIENT_BINARY"
if [ -n "$JAVA_PREFIX" ]
then
    echo "Fixing rpath for JAVA_PREFIX: $JAVA_PREFIX"
    JAVA_RPATH=$(find "$JAVA_PREFIX/lib" -name '*.so' | xargs -n 1 dirname | sort | uniq | grep -v jli | paste -sd ":" -)
    echo "New Java rpath: $JAVA_RPATH"
    RPATH_WITHOUT_JAVA=$(chrpath -l "$OUTPUT_CLIENT_BINARY" | sed "s#$OUTPUT_CLIENT_BINARY: RUNPATH=##g" | tr ':' '\n' | xargs -n 1 | grep -v java | paste -sd ":" -)
    echo "Original rpath without Java: $RPATH_WITHOUT_JAVA"
    RPATH="$JAVA_RPATH:$RPATH_WITHOUT_JAVA" #:/usr/lib/x86_64-linux-gnu/jni/
    chmod u+w "$OUTPUT_CLIENT_BINARY"
    chrpath -r "$RPATH" "$OUTPUT_CLIENT_BINARY"
    chmod u-w "$OUTPUT_CLIENT_BINARY"
fi