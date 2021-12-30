#!/bin/bash

#shellcheck disable=SC2068
bash build_local.sh $@ --config no-tty --config python --config java --config slow-wrapper-py3
