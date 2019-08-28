#!/bin/bash
UDFCLIENT_BIN=$1
$UDFCLIENT_BIN 2>&1 | tee exaudfclient_output
if grep -q 'Usage:' exaudfclient_output
then
    echo "Found usage output, this executable seems to be fine."
    rm exaudfclient_output
    exit 0
else
    echo "Could not find usage output, this might indicate that the executable does not work properly."
    rm exaudfclient_output
    exit 1
fi