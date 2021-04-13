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

echo ./install_via_pip.pl with empty
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/empty_test_file --python-binary python3 $DRY_RUN_OPTION
echo

echo ./install_via_pip.pl without versions
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/without_versions --python-binary python3 $DRY_RUN_OPTION
echo

echo ./install_via_pip.pl with versions, without allow-no-version
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/all_versions_specified --with-versions --python-binary python3 $DRY_RUN_OPTION
echo

echo ./install_via_pip.pl with versions, with allow-no-version, all versions specified
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/all_versions_specified --with-versions --allow-no-version --python-binary python3 $DRY_RUN_OPTION
echo

echo ./install_via_pip.pl with versions, with allow-no-version, some versions missing
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/some_missing_versions --with-versions --allow-no-version --python-binary python3 $DRY_RUN_OPTION
echo

echo ./install_via_pip.pl with versions, with allow-no-version-for-urls, file with urls
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/with_urls --with-versions --allow-no-version-for-urls --python-binary python3 $DRY_RUN_OPTION
echo

echo ./install_via_pip.pl with versions, with allow-no-version-for-urls, file with urls and some missing versions
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/with_urls_some_missing_versions --with-versions --allow-no-version-for-urls --python-binary python3 $DRY_RUN_OPTION || echo PASSED 
echo

echo ./install_via_pip.pl with pip version syntax
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/pip_version_syntax --python-binary python3 $DRY_RUN_OPTION
echo

echo "All pip tests passed"
