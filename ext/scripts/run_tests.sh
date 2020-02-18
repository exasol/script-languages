./install_batch.pl --file install_batch_test_file --element-separator ";;" --combining-template "echo 'install(c(<<<<0>>>>),c(<<<<1>>>>))'" --templates '"<<<<0>>>>"' ',' '"<<<<1>>>>"' ','
echo
./install_via_apt.pl --file install_via_apt_test_file --dry-run
echo
./install_via_pip.pl --file install_via_pip_test_file --python-binary python3 --dry-run
echo
./install_via_r_versions.pl --file install_via_r_versions_test_file --rscript-binary Rscript --dry-run
