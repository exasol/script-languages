#!/bin/bash

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PATH_TO_INSTALL_SCRIPTS="$SCRIPT_DIR/../../install_scripts"
DRY_RUN_OPTION=--dry-run
if [ "${1-}" == "--no-dry-run" ]
then
  DRY_RUN_OPTION=
fi

##############################################################################################################
#For this test we try to install Github CLI (https://cli.github.com/) as it is quite small (.deb has ~6MB) ##
##############################################################################################################
echo ./install_key.pl with Github CLI key
$PATH_TO_INSTALL_SCRIPTS/install_key.pl --key C99B11DEB97541F0 --key-server keyserver.ubuntu.com $DRY_RUN_OPTION
echo

echo ./install_ppa.pl with Github CLI repository
$PATH_TO_INSTALL_SCRIPTS//install_ppa.pl --ppa 'deb https://cli.github.com/packages bionic main' --out-file github $DRY_RUN_OPTION
echo

# Now we should be able to install gh (which is located in the github cli repository)
echo ./install_via_apt.pl with package gh
$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl --file test_files/apt/github_cli --with-versions $DRY_RUN_OPTION
echo

echo "All apt tests passed"
