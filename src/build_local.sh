export PYTHON_PREFIX=/usr 
export PYTHON_VERSION=python2.7
export CUSTOM_PROTOBUF_BIN=/usr/local/bin/protoc 
bazel build --subcommands --copt="-DCUSTOM_PROTOBUF_PREFIX=\"/usr/lib/x86_64-linux-gnu\"" --verbose_failures  $*