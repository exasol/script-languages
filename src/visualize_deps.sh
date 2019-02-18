export JAVA_PREFIX=/usr/lib/jvm/java-11-openjdk-amd64
export PYTHON2_PREFIX=/usr 
export PYTHON2_VERSION=python2
export PYTHON3_PREFIX=/usr 
export PYTHON3_VERSION=python3
export CUSTOM_PROTOBUF_BIN=/usr/local/bin/protoc 
export CUSTOM_PROTOBUF_PREFIX=/usr/lib/x86_64-linux-gnu
DEPS=""
for I in $1
do
    if [ "$DEPS" == "" ]
    then
        DEPS="deps($I)"
    else
        DEPS="$DEPS union deps($I)" 
    fi 
done
DEPS="($DEPS)"
shift 1
bazel query $* --noimplicit_deps "$DEPS except (filter('@bazel_tools', $DEPS) union filter('@local_config',$DEPS))" --output graph > graph.in
dot -Tpng < graph.in > graph.png