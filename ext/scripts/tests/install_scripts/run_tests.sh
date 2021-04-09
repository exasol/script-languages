#!/bin/bash

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PATH_TO_INSTALL_SCRIPTS="$SCRIPT_DIR/../../install_scripts"

echo install_batch.pl
$PATH_TO_INSTALL_SCRIPTS/install_batch.pl --file test_files/install_batch_test_file --element-separator ";;" --combining-template "echo 'install(c(<<<<0>>>>),c(<<<<1>>>>))'" --templates '"<<<<0>>>>"' ',' '"<<<<1>>>>"' ','
echo

echo Run Apt Tests
bash run_apt_tests.sh "$@"

echo Run Pip Tests
bash run_pip_tests.sh "$@"

# echo Run R versions Tests
# bash run_r_versions_tests.sh "$@"

echo Run R remotes Tests
bash run_r_remotes_tests.sh "$@"
