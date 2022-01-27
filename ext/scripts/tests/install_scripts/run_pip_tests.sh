#!/bin/bash

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# shellcheck source=./ext/scripts/tests/install_scripts/assert.sh
source "$SCRIPT_DIR/assert.sh"

PATH_TO_INSTALL_SCRIPTS="$SCRIPT_DIR/../../install_scripts"
DRY_RUN_OPTION=--dry-run
if [ "${1-}" == "--no-dry-run" ]
then
  DRY_RUN_OPTION=
fi

echo ./install_via_pip.pl with empty
TEST_OUTPUT=$("$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/empty_test_file --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" ""
echo

echo ./install_via_pip.pl without versions
TEST_OUTPUT=$("$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/without_versions --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly' 'requests' 'git+http://github.com/exasol/bucketfs-utils-python.git@main#egg=exasol-bucketfs-utils-python'"
echo

echo ./install_via_pip.pl without versions and --ignore-installed
TEST_OUTPUT=$("$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/without_versions --ignore-installed --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install --ignore-installed --no-cache-dir 'humanfriendly' 'requests' 'git+http://github.com/exasol/bucketfs-utils-python.git@main#egg=exasol-bucketfs-utils-python'"
echo

echo ./install_via_pip.pl without versions and --use-deprecated-legacy-resolver
TEST_OUTPUT=$("$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/without_versions --use-deprecated-legacy-resolver --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install --use-deprecated=legacy-resolver --no-cache-dir 'humanfriendly' 'requests' 'git+http://github.com/exasol/bucketfs-utils-python.git@main#egg=exasol-bucketfs-utils-python'"
echo

echo ./install_via_pip.pl with versions, without allow-no-version
TEST_OUTPUT=$("$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/all_versions_specified --with-versions --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests==2.21.0'"
echo

echo ./install_via_pip.pl with versions, with allow-no-version, all versions specified
TEST_OUTPUT=$("$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/all_versions_specified --with-versions --allow-no-version --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests==2.21.0'"
echo

echo ./install_via_pip.pl with versions, with allow-no-version, some versions missing
TEST_OUTPUT=$("$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/some_missing_versions --with-versions --allow-no-version --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests' 'git+http://github.com/exasol/bucketfs-utils-python.git@main#egg=exasol-bucketfs-utils-python'"
echo

echo ./install_via_pip.pl with versions, with allow-no-version-for-urls, file with urls
TEST_OUTPUT=$("$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/with_urls --with-versions --allow-no-version-for-urls --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests==2.27.1' 'git+http://github.com/exasol/bucketfs-utils-python.git@main#egg=exasol-bucketfs-utils-python'"
echo

echo ./install_via_pip.pl with versions, with allow-no-version-for-urls, file with urls and some missing versions
"$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/with_urls_some_missing_versions --with-versions --allow-no-version-for-urls --python-binary python3 "$DRY_RUN_OPTION" || echo PASSED
echo

echo ./install_via_pip.pl with pip version syntax
TEST_OUTPUT=$("$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/pip_version_syntax --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests>=2.21.0' 'git+http://github.com/exasol/bucketfs-utils-python.git@main#egg=exasol-bucketfs-utils-python'"
echo

check_for_failed_tests
echo "All pip tests passed"
