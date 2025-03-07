#!/bin/bash

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# shellcheck source=./ext/scripts/tests/install_scripts/assert.sh
source "$SCRIPT_DIR/assert.sh"
source "$SCRIPT_DIR/run_pip_tests_base.sh"

NL=$'\n'
echo ./install_via_pip.pl installing PIP package which require ephemeral installation of build tools
TEST_OUTPUT=$(run_install "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/needs_build_tools --python-binary python3 --install-build-tools-ephemerally $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: apt-get update && apt-get install -y build-essential pkg-config${NL}Dry-Run: python3 -m pip install  --no-cache-dir 'pybloomfiltermmap3'${NL}Dry-Run: apt-get purge -y build-essential pkg-config && apt-get -y autoremove"
echo

echo ./install_via_pip.pl installing PIP package which require ephemeral installation of build tools
TEST_OUTPUT=$(run_install_must_fail "$PATH_TO_INSTALL_SCRIPTS/install_via_pip.pl --file test_files/pip/needs_build_tools --python-binary python3 $DRY_RUN_OPTION")
assert "$TEST_OUTPUT" "Dry-Run: python3 -m pip install  --no-cache-dir 'pybloomfiltermmap3'"
echo


check_for_failed_tests
echo "All pip tests passed"
