DEPS=""
for I in "$1"
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