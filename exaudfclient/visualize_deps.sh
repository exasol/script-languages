#!/bin/bash

#Reason: .env file does not exist in repo. Need to disable shellcheck rule.
#shellcheck disable=SC1091
source .env
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
bazel query "$@" --noimplicit_deps "$DEPS except (filter('@bazel_tools', $DEPS) union filter('@local_config',$DEPS))" --output graph | sed -e "s/label/xlabel/g" > graph.in

dot -Grank=max   -Gsplines=ortho  -Goverlap=false -Granksep=4 -Gnodesep=2  -Tpng < graph.in > graph.png

# -Gconcentrate=true -Gsplines=ortho  -Nshape=box  -Gconcentrate=true