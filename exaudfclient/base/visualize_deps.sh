source .env
DEPS=""
for I in "$@"
do
    echo "I: $I"
    if [ "$DEPS" == "" ]
    then
        DEPS="deps($I)"
    else
        DEPS="$DEPS union deps($I)" 
    fi 
done
DEPS="($DEPS)"

exclude_filter=""
declare -a excludes=("@bazel_tools" "@local_config" "@ssl" "@java" "@python3" "@python2" "@protobuf" "@numpy")
for exclude in "${excludes[@]}"
do
    if [ "$exclude_filter" == "" ]
    then
        exclude_filter="filter('$exclude', $DEPS)"
    else
        exclude_filter="$exclude_filter union filter('$exclude', $DEPS)"
    fi
done

bazel query --noimplicit_deps --nohost_deps  "$DEPS except ($exclude_filter)" --output graph | sed -e "s/label/xlabel/g" > graph.in

dot -Tpng < graph.in > graph.png  

#dot -Grank=max   -Gsplines=ortho  -Goverlap=false -Granksep=4 -Gnodesep=2  -Tpng < graph.in > graph.png

# -Gconcentrate=true -Gsplines=ortho  -Nshape=box  -Gconcentrate=true
