#!/bin/bash

set -e
set -u
set -o pipefail

PATH_TO_INSTALL_SCRIPTS="../../install_scripts"

echo ./install_via_r_remotes.pl with empty
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/empty_test_file --dry-run --rscript-binary Rscript
echo

echo ./install_via_r_remotes.pl without versions
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/r/versions/without_versions --dry-run --rscript-binary Rscript 
echo

echo ./install_via_r_remotes.pl with versions, without allow-no-version
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/r/versions/with_versions/all_versions_specified --with-versions --dry-run --rscript-binary Rscript
echo

echo ./install_via_r_remotes.pl with versions, with allow-no-version, all versions specified
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/r/versions/with_versions/all_versions_specified --with-versions --allow-no-version --dry-run --rscript-binary Rscript
echo

echo ./install_via_r_remotes.pl with versions, with allow-no-version, some versions missing
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/r/versions/with_versions/some_missing_versions --with-versions --allow-no-version --dry-run --rscript-binary Rscript 
echo
