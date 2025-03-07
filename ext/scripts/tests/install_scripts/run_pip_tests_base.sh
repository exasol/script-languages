#!/bin/bash

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

if [ -z "${PATH_TO_INSTALL_SCRIPTS-}" ]
then
  PATH_TO_INSTALL_SCRIPTS="$SCRIPT_DIR/../../install_scripts"
fi

DRY_RUN_OPTION=--dry-run
if [ "${1-}" == "--no-dry-run" ]
then
  DRY_RUN_OPTION=
fi


if [ -z "${RUN_PIP_TESTS_EXECUTOR-}" ]
then
  echo Running pip tests without executor.
else
  echo Running pip tests with executor "'$RUN_PIP_TESTS_EXECUTOR'".
fi

function run_install() {
  if [ -z "${RUN_PIP_TESTS_EXECUTOR-}" ]
  then
    eval "$*"
  else
    eval "$RUN_PIP_TESTS_EXECUTOR $PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl $*"
  fi
}

function run_install_must_fail() {
  if [ -z "${RUN_PIP_TESTS_EXECUTOR-}" ]
  then
    if [ -z "${DRY_RUN_OPTION-}" ]
    then
      eval "$*"  && return 1 || return 0;
    else
      eval "$*"
    fi
  else
    if [ -z "${DRY_RUN_OPTION-}" ]
    then
      eval "$RUN_PIP_TESTS_EXECUTOR $PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl $*" && return 1 || return 0;
    else
      eval "$RUN_PIP_TESTS_EXECUTOR $PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl $*"
    fi
  fi
}
