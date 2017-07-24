#!/bin/bash
echo Building image
docker build --rm=true --tag exasol/script_lang_base_container .
echo Creating image
docker run --name exasol_script_lang_base_container exasol/script_lang_base_container 
docker stop exasol_script_lang_base_container
echo Exporting container
docker export exasol_script_lang_base_container > ScriptLanguages-6.0.2.tar
echo Cleanup Docker
docker rm exasol_script_lang_base_container
docker rmi exasol/script_lang_base_container
echo Preparing Container:
echo 1. Unpack
mkdir ScriptLanguages-6.0.2
tar xf ScriptLanguages-6.0.2.tar -C ScriptLanguages-6.0.2 --exclude=dev --exclude=proc
echo 2. Creating Dirs
mkdir ScriptLanguages-6.0.2/conf ScriptLanguages-6.0.2/proc ScriptLanguages-6.0.2/dev ScriptLanguages-6.0.2/exasol ScriptLanguages-6.0.2/buckets
echo 3. Creating links
rm ScriptLanguages-6.0.2/etc/resolv.conf ScriptLanguages-6.0.2/etc/hosts
ln -s /conf/resolv.conf ScriptLanguages-6.0.2/etc/resolv.conf
ln -s /conf/hosts ScriptLanguages-6.0.2/etc/hosts
echo 4. Package
cd ScriptLanguages-6.0.2
tar --numeric-owner --owner=0 --group=0 -zcf ../ScriptLanguages-6.0.2.tar.gz *
cd ..
echo 5. Cleanup temp files
rm ScriptLanguages-6.0.2.tar
rm -rf ScriptLanguages-6.0.2
echo Done.
