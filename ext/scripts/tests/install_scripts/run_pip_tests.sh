#!/bin/bash

set -e
set -u
set -o pipefail

PATH_TO_INSTALL_SCRIPTS="../../install_scripts"

echo ./install_via_pip.pl with empty
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/empty_test_file --dry-run --python-binary python3
echo

echo ./install_via_pip.pl without versions
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/without_versions --dry-run --python-binary python3 
echo

echo ./install_via_pip.pl with versions, without allow-no-version
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/all_versions_specified --with-versions --dry-run --python-binary python3
echo

echo ./install_via_pip.pl with versions, with allow-no-version, all versions specified
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/all_versions_specified --with-versions --allow-no-version --dry-run --python-binary python3
echo

echo ./install_via_pip.pl with versions, with allow-no-version, some versions missing
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/some_missing_versions --with-versions --allow-no-version --dry-run --python-binary python3 
echo

echo ./install_via_pip.pl with versions, with allow-no-version-for-urls, file with urls
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/with_urls --with-versions --allow-no-version-for-urls --dry-run --python-binary python3 
echo

echo ./install_via_pip.pl with versions, with allow-no-version-for-urls, file with urls and some missing versions
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/with_versions/with_urls_some_missing_versions --with-versions --allow-no-version-for-urls --dry-run --python-binary python3 || echo PASSED 
echo

echo ./install_via_pip.pl with pip version syntax
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/pip_version_syntax --dry-run --python-binary python3
echo

