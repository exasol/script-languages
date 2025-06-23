#!/bin/bash

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# shellcheck source=./ext/scripts/tests/install_scripts/assert.sh
source "$SCRIPT_DIR/assert.sh"
# shellcheck source=./ext/scripts/tests/install_scripts/run_pip_tests_base.sh
source "$SCRIPT_DIR/run_pip_tests_base.sh"

echo ./install_via_pip.pl with empty
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/empty_test_file --python-binary python3 "$DRY_RUN_OPTION")
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
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-common==1.1.4' 'azure-common==1.1.28'"
echo


echo ./install_via_pip.pl installing with ancestors but all empty
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/empty/step2 --ancestor-pip-package-root-path test_files/pip/empty/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" ""
echo

#Following tests verify the --ancestor-pip-package-root-path parameter
#However, the scenarios tested here are not exactly the same as in the real SLC builds, because we do not run multiple pip installations (and install the ancestors before) but run only the pip install command for the "current" build step.

echo ./install_via_pip.pl installing with ancestors and correct dependency
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/no_version_conflict/dependency_already_installed/step2 --ancestor-pip-package-root-path test_files/pip/no_version_conflict/dependency_already_installed/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-common==1.1.4' 'azure-batch==1.0.0'"
echo


echo ./install_via_pip.pl installing with ancestors and dependency with wrong version must fail
TEST_OUTPUT=$(run_install_must_fail "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/version_conflict/dependency_already_installed/step2 --ancestor-pip-package-root-path test_files/pip/version_conflict/dependency_already_installed/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-common==1.1.28' 'azure-batch==1.0.0'"
echo


echo ./install_via_pip.pl installing with ancestors and package which has a dependency to an older package must fail
TEST_OUTPUT=$(run_install_must_fail "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/version_conflict/other_package_with_older_dependency_already_installed/step2 --ancestor-pip-package-root-path test_files/pip/version_conflict/other_package_with_older_dependency_already_installed/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-storage-queue==1.1.0' 'azure-batch==1.0.0'"
echo


echo ./install_via_pip.pl installing with ancestors and package which has a dependency to a newer package must fail
TEST_OUTPUT=$(run_install_must_fail "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/version_conflict/other_package_with_newer_dependency_already_installed/step2 --ancestor-pip-package-root-path test_files/pip/version_conflict/other_package_with_newer_dependency_already_installed/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-batch==1.0.0' 'azure-storage-queue==1.1.0'"
echo


echo ./install_via_pip.pl installing with multiple ancestors
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/no_version_conflict/multiple_ancestors/step3 --ancestor-pip-package-root-path test_files/pip/no_version_conflict/multiple_ancestors/build_info/packages --python-binary python3 --with-versions $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'azure-common==1.1.28' 'azure-batch==14.2.0' 'azure-storage-queue==1.1.0'"
echo

echo ./install_via_pip.pl with pip-needs-break-system-packages
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl" --file test_files/pip/with_versions/all_versions_specified --pip-needs-break-system-packages --python-binary python3 "$DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install --break-system-packages --no-cache-dir 'humanfriendly' 'requests'"
echo

check_for_failed_tests
echo "All pip tests passed"
