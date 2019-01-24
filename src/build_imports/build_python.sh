function build_python{

    if [ "$ENABLE_PYTHON_IMPL" = "yes" ] || [ "$ENABLE_PYTHON3_IMPL" = "yes" ] ; then
        # Python
        echo "Copying Python related files to the build dir"
        for SRC in \
            pythoncontainer.cc exascript_python_preset.py exascript_python_wrap.py
        do
        cp "$SRC" "$BUILDDIR/" || die "Failed to copy file $SRC to build dir: $BUILDDIR."
        done
    fi

    if [ "$ENABLE_PYTHON_IMPL" = "yes" ]; then
        echo "Generating Python SWIG code"
        # create python wrapper from swig files
        swig -I${PYTHON_PREFIX}/include/python2.7 -O -DEXTERNAL_PROCESS -Wall -c++ -python -addextern -module exascript_python -o exascript_python_tmp.cc exascript.i || die "SWIG compilation failed."
        swig -I${PYTHON_PREFIX}/include/python2.7 -DEXTERNAL_PROCESS -c++ -python -external-runtime exascript_python_tmp.h || die "SWIG compilation failed."

        mv exascript_python_preset.py exascript_python_preset.py_orig
        echo "import sys, os" > exascript_python_preset.py
        
        echo "sys.path.extend($($PYTHON_PREFIX/bin/python -c 'import sys; import site; print sys.path'))" >> exascript_python_preset.py

        #echo "import sys, types, os;has_mfs = sys.version_info > (3, 5);p = os.path.join(sys._getframe(1).f_locals['sitedir'], *('google',));importlib = has_mfs and __import__('importlib.util');has_mfs and __import__('importlib.machinery');m = has_mfs and sys.modules.setdefault('google', importlib.util.module_from_spec(importlib.machinery.PathFinder.find_spec('google', [os.path.dirname(p)])));m = m or sys.modules.setdefault('google', types.ModuleType('google'));mp = (m or []) and m.__dict__.setdefault('__path__',[]);(p not in mp) and mp.append(p)" >> exascript_python_preset.py
        
        #echo "PyRun_String(\"import sys,os\", Py_single_input, globals, globals);" > generated_py_import.cc
        #echo "PyRun_String(\"sys.path.extend($($PYTHON_PREFIX/bin/python -c 'import sys; import site; print sys.path'))\",Py_single_input, globals, globals);" > generated_py_syspath.cc

    #    echo "sys.path.append('$PYTHON_PREFIX/lib/python2.7')" >> exascript_python_preset.py
    #    echo "sys.path.append('$PYTHON_PREFIX/lib/python2.7/site-packages')" >> exascript_python_preset.py
    #    echo "sys.path.append('$PYTHON_PREFIX/lib/python2.7/dist-packages')" >> exascript_python_preset.py
    #    echo "sys.path.append('$PYTHON_PREFIX/local/lib/python2.7')" >> exascript_python_preset.py
    #    echo "sys.path.append('$PYTHON_PREFIX/local/lib/python2.7/site-packages')" >> exascript_python_preset.py
    #    echo "sys.path.append('$PYTHON_PREFIX/local/lib/python2.7/dist-packages')" >> exascript_python_preset.py


        if [ ! "X$PYTHON_SYSPATH" = "X" ]; then
            echo "sys.path.extend($PYTHON_SYSPATH)" >> exascript_python_preset.py
        fi
        
        cat exascript_python_preset.py_orig >> exascript_python_preset.py
        
        python ./build_integrated.py exascript_python_int.h exascript_python.py exascript_python_wrap.py exascript_python_preset.py || die "Failed build_integrated"
        python ./filter_swig_code.py exascript_python.h exascript_python_tmp.h || die "Failed: filter_swig_code.py exascript_python.h exascript_python_tmp.h"
        python ./filter_swig_code.py exascript_python.cc exascript_python_tmp.cc || die "exascript_python.cc exascript_python_tmp.cc"

        CXXFLAGS="-DENABLE_PYTHON_VM -I$PYTHON_PREFIX/include/python2.7 $CXXFLAGS"
        LIBS="-lpython2.7 $LIBS"
        LDFLAGS="-L$PYTHON_PREFIX/lib -Wl,-rpath,$PYTHON_PREFIX/lib $LDFLAGS" 

        echo "Compiling Python specific code"
        g++ -o exascript_python.o -c exascript_python.cc $CXXFLAGS -Wno-unused-but-set-variable || die "Failed to compile exascript_python.o"
        g++ -o pythoncontainer.o -c pythoncontainer.cc $CXXFLAGS || die "Failed to compile pythoncontainer.o"

        CONTAINER_CLIENT_OBJECT_FILES="exascript_python.o pythoncontainer.o $CONTAINER_CLIENT_OBJECT_FILES"
    fi

    if [ "$ENABLE_PYTHON3_IMPL" = "yes" ]; then
        PYTHON3_VERSION="python3.6"
        PYTHON3_CONFIG="python3-config"
        hash $PYTHON3_VERSION-config && PYTHON3_CONFIG="$PYTHON3_VERSION-config"

        echo "Generating Python3 SWIG code using python3-config: $PYTHON3_CONFIG"
        # create python wrapper from swig files
        swig $($PYTHON3_CONFIG --includes) -O -DEXTERNAL_PROCESS -Wall -c++ -python -py3 -addextern -module exascript_python -o exascript_python_tmp.cc exascript.i || die "SWIG compilation failed."
        swig $($PYTHON3_CONFIG --includes) -DEXTERNAL_PROCESS -c++ -python -py3 -external-runtime exascript_python_tmp.h || die "SWIG compilation failed."

        mv exascript_python_preset.py exascript_python_preset.py_orig
        echo "import sys, os" > exascript_python_preset.py
        
        echo "sys.path.extend($($PYTHON_PREFIX/bin/python3 -c 'import sys; import site; print(sys.path)'))" >> exascript_python_preset.py

        if [ ! "X$PYTHON_SYSPATH" = "X" ]; then
            echo "sys.path.extend($PYTHON_SYSPATH)" >> exascript_python_preset.py
        fi
        
        cat exascript_python_preset.py_orig >> exascript_python_preset.py
        
        python ./build_integrated.py exascript_python_int.h exascript_python.py exascript_python_wrap.py exascript_python_preset.py || die "Failed build_integrated"
        cp exascript_python_tmp.h exascript_python.h || die "Failed: filter_swig_code.py exascript_python.h exascript_python_tmp.h"
        cp exascript_python_tmp.cc exascript_python.cc || die "exascript_python.cc exascript_python_tmp.cc"

        CXXFLAGS="-DENABLE_PYTHON_VM -DENABLE_PYTHON3 $($PYTHON3_CONFIG --includes) $CXXFLAGS"
        LIBS="$($PYTHON3_CONFIG --libs) $LIBS"
        LDFLAGS="-L$($PYTHON3_CONFIG --prefix)/lib -Wl,-rpath,$($PYTHON3_CONFIG --prefix)/lib $LDFLAGS" 

        echo "Compiling Python3 specific code with these CXXFLAGS:$CXXFLAGS"
        g++ -o exascript_python.o -c exascript_python.cc $CXXFLAGS -Wno-unused-but-set-variable || die "Failed to compile exascript_python.o"
        g++ -o pythoncontainer.o -c pythoncontainer.cc $CXXFLAGS || die "Failed to compile pythoncontainer.o"
        g++ -shared $CXXFLAGS -I/usr/local/lib/$PYTHON3_VERSION/dist-packages/numpy/core/include $($PYTHON3_CONFIG --libs) -opyextdataframe.so python_ext_dataframe.cc || die "Failed to compile pyextdataframe.so"

        CONTAINER_CLIENT_OBJECT_FILES="exascript_python.o pythoncontainer.o $CONTAINER_CLIENT_OBJECT_FILES"
    fi

}