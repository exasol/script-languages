function build_r{

    if [ "$ENABLE_R_IMPL" = "yes" ]; then
        # R
        echo "Copying R related files to the build dir"
        for SRC in \
            rcontainer.cc exascript_r_preset.R exascript_r_wrap.R
        do
        cp "$SRC" "$BUILDDIR/" || die "Failed to copy file $SRC to build dir: $BUILDDIR."
        done
    fi

    if [ "$ENABLE_R_IMPL" = "yes" ]; then
        # create R wrapper from swig files
        swig -O -DEXTERNAL_PROCESS -Wall -c++ -r -addextern -module exascript_r -o exascript_r_tmp.cc exascript.i >/dev/null 2>&1 || die "SWIG compilation failed."
        swig -DEXTERNAL_PROCESS -c++ -r -external-runtime exascript_r_tmp.h || die "SWIG compilation failed."


        python ./build_integrated.py exascript_r_int.h exascript_r.R exascript_r_wrap.R exascript_r_preset.R 
        python ./filter_swig_code.py exascript_r.h exascript_r_tmp.h
        python ./filter_swig_code.py exascript_r.cc exascript_r_tmp.cc

        LIBS="-lR $LIBS"
        CXXFLAGS="-DENABLE_R_VM -I/usr/share/R/include $CXXFLAGS"

        g++ -o exascript_r.o -c exascript_r.cc $CXXFLAGS || die "Failed to compile exascript_r.o"
        g++ -o rcontainer.o -c rcontainer.cc $CXXFLAGS || die "Failed to compile rcontainer.o"

        CONTAINER_CLIENT_OBJECT_FILES="exascript_r.o rcontainer.o $CONTAINER_CLIENT_OBJECT_FILES"
    fi
}