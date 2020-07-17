#!/bin/bash

set -e
set -u
set -o pipefail

PATH_TO_INSTALL_SCRIPTS="../../install_scripts"

echo ./install_via_apt.pl with empty
$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl --file test_files/empty_test_file --with-versions --dry-run
echo

echo ./install_via_apt.pl without versions
$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl --file test_files/apt/without_versions --dry-run
echo

echo ./install_via_apt.pl with versions, without allow-no-version
$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl --file test_files/apt/with_versions/all_versions_specified --with-versions --dry-run
echo

echo ./install_via_apt.pl with versions, with allow-no-version, all versions specified
$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl --file test_files/apt/with_versions/all_versions_specified --with-versions --allow-no-version --dry-run 
echo

echo ./install_via_apt.pl with versions, with allow-no-version, some versions missing
$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl --file test_files/apt/with_versions/some_missing_versions --with-versions --allow-no-version --dry-run 
echo

