#!/bin/bash

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# shellcheck source=./ext/scripts/tests/install_scripts/assert.sh
source "$SCRIPT_DIR/assert.sh"

if [ -z "${PATH_TO_INSTALL_SCRIPTS-}" ]
then
  PATH_TO_INSTALL_SCRIPTS="$SCRIPT_DIR/../../install_scripts"
fi

DRY_RUN_OPTION=--dry-run
if [ "${1-}" == "--no-dry-run" ]
then
  DRY_RUN_OPTION=
fi

function run_install() {
  if [ -z "${RUN_PIP_TESTS_EXECUTOR-}" ]
  then
    eval "$*"
  else
    eval "$RUN_PIP_TESTS_EXECUTOR $PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl $*"
  fi
}

function run_install_must_fail() {
  if [ -z "${RUN_PIP_TESTS_EXECUTOR-}" ]
  then
    if [ -z "${DRY_RUN_OPTION-}" ]
    then
      eval "$*"  && return 1 || return 0;
    else
      eval "$*"
    fi
  else
    if [ -z "${DRY_RUN_OPTION-}" ]
    then
      eval "$RUN_PIP_TESTS_EXECUTOR $PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl $*" && return 1 || return 0;
    else
      eval "$RUN_PIP_TESTS_EXECUTOR $PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl $*"
    fi
  fi
}


echo ./install_via_pip.pl with empty
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/empty_test_file --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" ""
echo

echo ./install_via_pip.pl without versions
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/without_versions --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly' 'requests' 'git+http://github.com/exasol/bucketfs-utils-python.git@0.2.0#egg=exasol-bucketfs-utils-python'"
echo


echo ./install_via_pip.pl without versions and --ignore-installed
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/without_versions --ignore-installed --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install --ignore-installed --no-cache-dir 'humanfriendly' 'requests' 'git+http://github.com/exasol/bucketfs-utils-python.git@0.2.0#egg=exasol-bucketfs-utils-python'"
echo


echo ./install_via_pip.pl without versions and --use-deprecated-legacy-resolver
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/without_versions --use-deprecated-legacy-resolver --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install --use-deprecated=legacy-resolver --no-cache-dir 'humanfriendly' 'requests' 'git+http://github.com/exasol/bucketfs-utils-python.git@0.2.0#egg=exasol-bucketfs-utils-python'"
echo


echo ./install_via_pip.pl with versions, without allow-no-version
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/all_versions_specified --with-versions --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests==2.21.0'"
echo


echo ./install_via_pip.pl with versions, with allow-no-version, all versions specified
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/all_versions_specified --with-versions --allow-no-version --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests==2.21.0'"
echo


echo ./install_via_pip.pl with versions, with allow-no-version, some versions missing
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/some_missing_versions --with-versions --allow-no-version --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests' 'git+http://github.com/exasol/bucketfs-utils-python.git@0.2.0#egg=exasol-bucketfs-utils-python'"
echo


echo ./install_via_pip.pl with versions, with allow-no-version-for-urls, file with urls
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/with_urls --with-versions --allow-no-version-for-urls --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests==2.27.1' 'git+http://github.com/exasol/bucketfs-utils-python.git@0.2.0#egg=exasol-bucketfs-utils-python'"
echo


echo ./install_via_pip.pl with versions, with allow-no-version-for-urls, file with urls and some missing versions
run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/with_urls_some_missing_versions --with-versions --allow-no-version-for-urls --python-binary python3 "$DRY_RUN_OPTION" || echo PASSED
echo


echo ./install_via_pip.pl with pip version syntax
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/pip_version_syntax --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'humanfriendly==9.1' 'requests>=2.21.0' 'git+http://github.com/exasol/bucketfs-utils-python.git@0.2.0#egg=exasol-bucketfs-utils-python'"
echo


echo ./install_via_pip.pl installing a package twice with different versions must fail
TEST_OUTPUT=$(run_install_must_fail "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/version_conflict/same_pkg/step2 --ancestor-pip-package-root-path test_files/pip/version_conflict/same_pkg/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-common==1.1.28' 'azure-common==1.1.4'"
echo


echo ./install_via_pip.pl installing with ancestors but all empty
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/empty/step2 --ancestor-pip-package-root-path test_files/pip/empty/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" ""
echo


echo ./install_via_pip.pl installing with ancestors and correct dependency
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/no_version_conflict/dependency_already_installed/step2 --ancestor-pip-package-root-path test_files/pip/no_version_conflict/dependency_already_installed/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-batch==1.0.0' 'azure-common==1.1.4'"
echo


echo ./install_via_pip.pl installing with ancestors and dependency with wrong version must fail
TEST_OUTPUT=$(run_install_must_fail "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/version_conflict/dependency_already_installed/step2 --ancestor-pip-package-root-path test_files/pip/version_conflict/dependency_already_installed/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-batch==1.0.0' 'azure-common==1.1.28'"
echo


echo ./install_via_pip.pl installing with ancestors and package which has a dependency to an older package must fail
TEST_OUTPUT=$(run_install_must_fail "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/version_conflict/other_package_with_older_dependency_already_installed/step2 --ancestor-pip-package-root-path test_files/pip/version_conflict/other_package_with_older_dependency_already_installed/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-batch==1.0.0' 'azure-storage-queue==1.1.0'"
echo


echo ./install_via_pip.pl installing with ancestors and package which has a dependency to a newer package must fail
TEST_OUTPUT=$(run_install_must_fail "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/version_conflict/other_package_with_newer_dependency_already_installed/step2 --ancestor-pip-package-root-path test_files/pip/version_conflict/other_package_with_newer_dependency_already_installed/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-batch==1.0.0' 'azure-storage-queue==1.1.0'"
echo


echo ./install_via_pip.pl installing with multiple ancestors
TEST_OUTPUT=$(run_install_must_fail "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/no_version_conflict/multiple_ancestors/step3 --ancestor-pip-package-root-path test_files/pip/no_version_conflict/multiple_ancestors/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-batch==14.2.0' 'azure-common==1.1.28' 'azure-storage-queue==1.1.0'"
echo

check_for_failed_tests
echo "All pip tests passed"
