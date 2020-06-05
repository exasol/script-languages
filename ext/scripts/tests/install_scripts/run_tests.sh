#!/bin/bash

set -e
set -u
set -o pipefail

PATH_TO_INSTALL_SCRIPTS="../../"

echo install_batch.pl
$PATH_TO_INSTALL_SCRIPTS/install_batch.pl --file install_batch_test_file --element-separator ";;" --combining-template "echo 'install(c(<<<<0>>>>),c(<<<<1>>>>))'" --templates '"<<<<0>>>>"' ',' '"<<<<1>>>>"' ','
echo

echo ./install_via_apt.pl without versions
$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl --file install_via_apt_wo_versions_test_file --dry-run
echo

echo ./install_via_apt.pl with versions
$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl --file install_via_apt_with_versions_test_file --with-versions --dry-run
echo

echo ./install_via_apt.pl with empty
$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl --file install_empty_test_file --with-versions --dry-run
echo

echo ./install_via_pip.pl with version seperator, without versions
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file install_via_pip_test_file_with_version_seperator --python-binary python3 --dry-run
echo

echo ./install_via_pip.pl with version seperator, with versions, without allow-no-version
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file install_via_pip_test_file_with_version_seperator --python-binary python3 --with-versions --dry-run
echo

echo ./install_via_pip.pl with version seperator, with versions, with allow-no-version
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file install_via_pip_test_file_with_version_seperator --python-binary python3 --with-versions --allow-no-version --dry-run
echo

echo ./install_via_pip.pl with pip version syntax
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file install_via_pip_test_file_with_pip_version_syntax --python-binary python3 --dry-run
echo

echo ./install_via_pip.pl with empty
$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file install_empty_test_file --python-binary python3 --dry-run
echo

echo ./install_via_r_versions.pl
$PATH_TO_INSTALL_SCRIPTS/install_via_r_versions.pl --file install_via_r_versions_test_file --rscript-binary Rscript --dry-run
echo

echo ./install_via_r_versions.pl with empty
$PATH_TO_INSTALL_SCRIPTS/install_via_r_versions.pl --file install_empty_test_file --rscript-binary Rscript --dry-run
