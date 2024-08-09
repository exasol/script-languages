#!/bin/bash

bash build_local.sh "$@" --lockfile_mode=off --config no-tty --config python --config java --config slow-wrapper-py3
