#!/bin/bash
set -e
set -u
set -o pipefail
set -x

DRIVER_VERSION=$1
pushd /tmp
# download driver
DRIVER_NAME=NVIDIA-Linux-x86_64-$DRIVER_VERSION
curl -o "$DRIVER_NAME.run"  "http://us.download.nvidia.com/tesla/$DRIVER_VERSION/$DRIVER_NAME.run"
# extract libs
chmod +x "$DRIVER_NAME.run"
./"$DRIVER_NAME.run" -x

pushd "$DRIVER_NAME"
# copy libs
LIB_DIR="/opt/cuda/lib"
mkdir -p "$LIB_DIR"
cp -R {libcuda,libnvidia}*so* "$LIB_DIR"
cp -R {libnvcuvid,libnvoptix,libvdpau_nvidia,libOpenCL,libOpenGL}.so* "$LIB_DIR"
# copy binaries
BIN_DIR=/opt/cuda/bin
mkdir -p "$BIN_DIR"
cp -R nvidia-smi "$BIN_DIR"
popd

popd

pushd "$LIB_DIR"
for lib_file in *"$DRIVER_VERSION"; do
    lib_file_base="${lib_file%.$DRIVER_VERSION}"
    ln -sf "$lib_file" "$lib_file_base.1"
    ln -sf "$lib_file_base.1" "$lib_file_base"
done
echo "$LIB_DIR" > /etc/ld.so.conf.d/cuda.conf
popd
ldconfig


