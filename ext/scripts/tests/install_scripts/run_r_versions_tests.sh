#!/bin/bash

set -e
set -u
set -o pipefail

PATH_TO_INSTALL_SCRIPTS="../../install_scripts"

echo ./install_via_r_versions.pl
$PATH_TO_INSTALL_SCRIPTS/install_via_r_versions.pl --file test_files/r/versions/with_versions/all_versions_specified --rscript-binary Rscript --dry-run
echo

echo ./install_via_r_versions.pl with empty
$PATH_TO_INSTALL_SCRIPTS/install_via_r_versions.pl --file test_files/empty_test_file --dry-run --rscript-binary Rscript
