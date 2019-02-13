export JAVA_PREFIX=/usr/lib/jvm/java-11-openjdk-amd64
export PYTHON_PREFIX=/usr 
export PYTHON_VERSION=python3
export CUSTOM_PROTOBUF_BIN=/usr/local/bin/protoc 
export CUSTOM_PROTOBUF_PREFIX=/usr/lib/x86_64-linux-gnu
export VERBOSE_BUILD="--subcommands --verbose_failures"
#export VERBOSE_BUILD=""
bash build.sh $*