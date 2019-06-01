#!/usr/bin/env bash
pipenv install
export PYTHONPATH="."
pipenv run python exaslct_src/test/$1