function build_streaming{

    if [ "$ENABLE_STREAMING_IMPL" = "yes" ]; then
        # Python
        echo "Copying Streaming related files to the build dir"
        for SRC in \
            streamingcontainer.cc
        do
            cp "$SRC" "$BUILDDIR/" || die "Failed to copy file $SRC to build dir: $BUILDDIR."
        done
    fi

    if [ "$ENABLE_STREAMING_IMPL" = "yes" ]; then

        CXXFLAGS="-DENABLE_STREAMING_VM $CXXFLAGS"

        echo "Compiling Streaming specific code"
        g++ -o streamingcontainer.o -c streamingcontainer.cc $CXXFLAGS || die "Failed to compile streamingcontainer.o"

        CONTAINER_CLIENT_OBJECT_FILES="streamingcontainer.o $CONTAINER_CLIENT_OBJECT_FILES"
    fi

}