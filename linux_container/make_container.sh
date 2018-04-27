#!/bin/bash
DOCKER="sudo docker"
VERSION_NAME="ScriptLanguages-"$(date --iso-860)
echo Building image
$DOCKER build --rm=true --tag exasol/script_lang_base_container .
echo Creating image
$DOCKER run --name exasol_script_lang_base_container exasol/script_lang_base_container 
$DOCKER stop exasol_script_lang_base_container
echo Exporting container
$DOCKER export exasol_script_lang_base_container > $VERSION_NAME.tar
echo Cleanup Docker
#$DOCKER rm exasol_script_lang_base_container
#$DOCKER rmi exasol/script_lang_base_container
echo Preparing Container:
echo 1. Unpack
mkdir $VERSION_NAME
tar xf $VERSION_NAME.tar -C $VERSION_NAME --exclude=dev --exclude=proc
echo 2. Creating Dirs
mkdir $VERSION_NAME/conf $VERSION_NAME/proc $VERSION_NAME/dev $VERSION_NAME/exasol $VERSION_NAME/buckets
echo 3. Creating links
rm $VERSION_NAME/etc/resolv.conf $VERSION_NAME/etc/hosts
ln -s /conf/resolv.conf $VERSION_NAME/etc/resolv.conf
ln -s /conf/hosts $VERSION_NAME/etc/hosts
echo 4. Package
cd $VERSION_NAME
tar --numeric-owner --owner=0 --group=0 -zcf ../$VERSION_NAME.tar.gz *
cd ..
echo 5. Cleanup temp files
rm $VERSION_NAME.tar
rm -rf $VERSION_NAME
echo Done.
