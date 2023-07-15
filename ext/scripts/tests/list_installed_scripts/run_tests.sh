#!/bin/bash

set -e
set -u
set -o pipefail

PATH_TO_INSTALL_SCRIPTS="../../list_installed_scripts"

echo list_installed_apt.sh
$PATH_TO_INSTALL_SCRIPTS/list_installed_apt.sh
echo

echo list_installed_pip.sh python3
$PATH_TO_INSTALL_SCRIPTS/list_installed_pip.sh python3
echo

echo list_installed_R.sh
$PATH_TO_INSTALL_SCRIPTS/list_installed_R.sh
echo
