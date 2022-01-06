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

echo ./install_via_apt.pl with empty
"$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl" --file test_files/empty_test_file --with-versions "$DRY_RUN_OPTION"
echo

echo ./install_via_apt.pl without versions
"$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl" --file test_files/apt/without_versions "$DRY_RUN_OPTION"
echo

echo ./install_via_apt.pl with versions, without allow-no-version
"$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl" --file test_files/apt/with_versions/all_versions_specified --with-versions "$DRY_RUN_OPTION"
echo

echo ./install_via_apt.pl with versions, with allow-no-version, all versions specified
"$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl" --file test_files/apt/with_versions/all_versions_specified --with-versions --allow-no-version "$DRY_RUN_OPTION"
echo

echo ./install_via_apt.pl with versions, with allow-no-version, some versions missing
"$PATH_TO_INSTALL_SCRIPTS/install_via_apt.pl" --file test_files/apt/with_versions/some_missing_versions --with-versions --allow-no-version "$DRY_RUN_OPTION"
echo
echo "All apt tests passed"
