#!/usr/bin/env bash

# Setup directory "exa" with pre-configured EXAConf to attach it to the exasoldb docker container
mkdir -p /exa/{etc,data/storage}
cp EXAConf /exa/etc/EXAConf
dd if=/dev/zero of=/exa/data/storage/dev.1.data bs=1 count=1 seek=4294967296
touch /exa/data/storage/dev.1.meta