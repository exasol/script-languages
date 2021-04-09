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

echo ./install_via_r_remotes.pl with empty
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/empty_test_file --rscript-binary Rscrip $DRY_RUN_OPTION
echo

echo ./install_via_r_remotes.pl without versions
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/r/versions/without_versions --rscript-binary Rscript $DRY_RUN_OPTION
echo

echo ./install_via_r_remotes.pl with versions, without allow-no-version
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/r/versions/with_versions/all_versions_specified --with-versions --rscript-binary Rscript $DRY_RUN_OPTION
echo

echo ./install_via_r_remotes.pl with versions, with allow-no-version, all versions specified
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/r/versions/with_versions/all_versions_specified --with-versions --allow-no-version --rscript-binary Rscript $DRY_RUN_OPTION
echo

echo ./install_via_r_remotes.pl with versions, with allow-no-version, some versions missing
$PATH_TO_INSTALL_SCRIPTS/install_via_r_remotes.pl --file test_files/r/versions/with_versions/some_missing_versions --with-versions --allow-no-version --rscript-binary Rscript $DRY_RUN_OPTION
echo

echo "All r remotes tests passed"
