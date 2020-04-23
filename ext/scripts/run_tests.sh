echo install_batch.pl
./install_batch.pl --file install_batch_test_file --element-separator ";;" --combining-template "echo 'install(c(<<<<0>>>>),c(<<<<1>>>>))'" --templates '"<<<<0>>>>"' ',' '"<<<<1>>>>"' ','
echo
echo ./install_via_apt.pl without versions
./install_via_apt.pl --file install_via_apt_wo_versions_test_file --dry-run
echo
echo ./install_via_apt.pl with versions
./install_via_apt.pl --file install_via_apt_with_versions_test_file --with-versions --dry-run
echo
echo ./install_via_apt.pl with empty
./install_via_apt.pl --file install_via_apt_empty --with-versions --dry-run
echo
echo ./install_via_pip.pl
./install_via_pip.pl --file install_via_pip_test_file --python-binary python3 --dry-run
echo
echo ./install_via_pip.pl with empty
./install_via_pip.pl --file install_via_apt_empty --python-binary python3 --dry-run
echo
echo ./install_via_r_versions.pl
./install_via_r_versions.pl --file install_via_r_versions_test_file --rscript-binary Rscript --dry-run
echo
echo ./install_via_r_versions.pl with empty
./install_via_r_versions.pl --file install_via_apt_empty --rscript-binary Rscript --dry-run
