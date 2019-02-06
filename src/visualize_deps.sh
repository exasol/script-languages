DEPS="deps($1)"
shift 1
for I in $*
do
    DEPS="$DEPS union deps($I)" 
done
DEPS="($DEPS)"
bazel query --noimplicit_deps "$DEPS except (filter('@bazel_tools', $DEPS) union filter('@local_config',$DEPS))" --output graph > graph.in
dot -Tpng < graph.in > graph.png