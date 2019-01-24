function parse_args {

    optarr=$(getopt -o 'h' --long 'help,src-dir:,build-dir:,output-dir:,enable-r,enable-java,enable-python,enable-python3,python-prefix:,python-syspath:,enable-streaming,custom-protobuf-prefix:,release-sources,java-flags:' -- "$@")

    eval set -- "$optarr"

    while true; do
        case "$1" in
            --src-dir) SRCDIR="$2"; shift 2;;
            --build-dir) BUILDDIR="$2"; shift 2;;
            --output-dir) OUTPUTDIR="$2"; shift 2;;
            --enable-python) ENABLE_PYTHON_IMPL="yes"; shift 1;;
        --enable-python3) ENABLE_PYTHON3_IMPL="yes"; shift 1;;
            --python-prefix) PYTHON_PREFIX="$2"; shift 2;;
            --python-syspath) PYTHON_SYSPATH="$2"; shift 2;;
            --enable-r) ENABLE_R_IMPL="yes"; shift 1;;        
            --enable-java) ENABLE_JAVA_IMPL="yes"; shift 1;;
        --java-flags) JAVA_FLAGS="$2"; shift 2;;
            --enable-streaming) ENABLE_STREAMING_IMPL="yes"; shift 1;;
            --custom-protobuf-prefix) CUSTOM_PROTOBUF_PREFIX="$2"; shift 2;;
        --release-sources) RELEASE_SOURCES="yes"; shift 1;;
            -h|--help) echo "Usage: $0 --src-dir=<dir> --build-dir=<dir> --output-dir=<>"
            echo "Options:"
            echo "  [--enable-python]   Enable support for the Python language in the script language client"
            echo "  [--enable-r]        Enable support for the R language in the script language client"
            echo "  [--enable-java]     Enable support for the Java language in the script language client"
            echo "  [-h|--help]         Print this help."; exit 0;;
            --) shift; break;;
            *) echo "Internal error while parsing arguments. ($1)"; exit 1;;
        esac
    done

    LD_LIBRARY_PATH=${PYTHON_PREFIX}/lib:${LD_LIBRARY_PATH}

    die() { echo "ERROR:" "$@" >&2; exit 1; }

    [ X"$SRCDIR" = X"" ] && die "Missing mandatory argument --src-dir"
    [ X"$BUILDDIR" = X"" ] && die "Missing mandatory argument --build-dir"
    [ X"$OUTPUTDIR" = X"" ] && die "Missing mandatory argument --output-dir"
    [ -d "$SRCDIR" ] || die "Directory specified via --src-dir does not exist: $SRCDIR"
    [ -d "$BUILDDIR" ] || die "Directory specified via --build-dir does not exist: $BUILDDIR"
    [ -d "$OUTPUTDIR" ] || die "Directory specified via --output-dir does not exist: $OUTPUTDIR"

}

function prepare_build_dir{
    cd $SRCDIR || die "Cannot change the current directory to $SRCDIR"

    mkdir -p  "$BUILDDIR"  "$BUILDDIR/java_src/com/exasol" || die "Failed to create directories in $BUILDDIR"

    export PATH=$PYTHON_PREFIX/bin:$PATH

    # Copy source code to the build dir
    echo "Copying common source files to the build dir"
    for SRC in \
            zmqcontainer.proto exascript.i filter_swig_code.py build_integrated.py exaudfclient.cc exaudflib* \
            script_data_transfer_objects* LICENSE-exasol-script-api.txt scriptoptionlines.h scriptoptionlines.cc \
            python_ext_dataframe.cc
    do
        cp "$SRC" "$BUILDDIR/" || die "Failed to copy file $SRC to build dir: $BUILDDIR."
    done
}

function create_source_from_proto{

    cd $BUILDDIR || die "No $BUILDDIR directory found"
    # create source code from proto files
    if [ "X$CUSTOM_PROTOBUF_PREFIX" = "X" ]; then
        protoc -I. zmqcontainer.proto --cpp_out=. || die "Failed to create C++ proto files."
    else
        $CUSTOM_PROTOBUF_PREFIX/bin/protoc -I. zmqcontainer.proto --cpp_out=. || die "Failed to create C++ proto files."
    fi
    #/usr/local/protoc zmqcontainer.proto --python_out=. || die "Failed to create Python proto files."
}

function compile_exaudfclient{
    # compile exaudfclient
    CXXFLAGS="-fPIC $CXXFLAGS"

    g++ -o scriptoptionlines.o -c scriptoptionlines.cc $CXXFLAGS || die "Failed to compile scriptoptionlines.o"

    if [ ! -z "$CUSTOM_PROTOBUF_PREFIX" ]; then
        CXXFLAGS="-DCUSTOM_PROTOBUF_PREFIX=\"$CUSTOM_PROTOBUF_PREFIX\" $CXXFLAGS"
    fi

    echo "================================================"
    echo "================================================"
    echo "= compiling exaudfclient.cc with"
    echo "= CXXFLAGS=$CXXFLAGS"
    echo "= LDFLAGS=$LDFLAGS"
    echo "================================================"
    echo "================================================"
    echo "================================================"

    g++ -o exaudfclient.o -c exaudfclient.cc $CXXFLAGS || die "Failed to compile exaudfclient.o"
    g++ -o zmqcontainer.pb.o -c zmqcontainer.pb.cc $CXXFLAGS || die "Failed to compile zmqcontainer.pb.o"

    g++ -o scriptDTOWrapper.o -c script_data_transfer_objects_wrapper.cc $CXXFLAGS || die "Failed to compile scriptDTOWrapper.o"
    g++ -o scriptDTO.o -c script_data_transfer_objects.cc $CXXFLAGS || die "Failed to compile scriptDTO.o"

    g++ -o exaudflib.o -c exaudflib.cc $CXXFLAGS || die "Failed to compile exaudflib.o"

    g++ -shared -o libexaudflib.so exaudflib.o zmqcontainer.pb.o scriptDTOWrapper.o scriptDTO.o -Wl,--no-as-needed -l zmq -g



    echo "g++ -o exaudfclient exaudfclient.o $CONTAINER_CLIENT_OBJECT_FILES scriptoptionlines.o -Wl,--no-as-needed scriptDTOWrapper.o scriptDTO.o $LDFLAGS $LIBS -g"

    g++ -o exaudfclient exaudfclient.o $CONTAINER_CLIENT_OBJECT_FILES scriptoptionlines.o -Wl,--no-as-needed scriptDTOWrapper.o scriptDTO.o $LDFLAGS $LIBS -g || die "Failed to compile exaudfclient"
}

function create_general_output_files{

    # Create output files
    cp -a "$BUILDDIR/exaudfclient" "$OUTPUTDIR/exaudfclient" || die "Failed to create $OUTPUTDIR/exaudfclient"
    cp -a "$BUILDDIR/libexaudflib.so" "$OUTPUTDIR/libexaudflib.so" || die "Failed to create $OUTPUTDIR/libexaudflib.so"
    chmod +x "$OUTPUTDIR/exaudfclient" || die "Failed chmod of $OUTPUTDIR/exaudfclient"

    if [ "$RELEASE_SOURCES" = "yes" ]; then
        sources_target="$OUTPUTDIR/src"
        mkdir -p $sources_target
        for SRC in \
            zmqcontainer.proto exaudfclient.cc exaudflib.cc exaudflib.h
        do
        cp "$BUILDDIR/$SRC" "$sources_target/" || die "Failed to copy file $SRC from $BUILDDIR to $sources_target"
        done
    fi
}
