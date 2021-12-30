#!/bin/bash

mkfifo /tmp/fff
INPUT=exasol_emulator/input.csv
SCRIPT=exasol_emulator/script.py
LANGUAGE=python
python exasol_emulator/exasolution.py unix:/tmp/fff $INPUT exasol_emulator/output.csv $SCRIPT &
"$1/exaudfclient" ipc:///tmp/fff lang=$LANGUAGE
