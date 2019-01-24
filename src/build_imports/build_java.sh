function build_java{

    if [ "$ENABLE_JAVA_IMPL" = "yes" ]; then
        # Java
        echo "Copying Java related files to the build dir"
        for SRC in \
            javacontainer.cc exascript_java_jni_decl.h
        do
        cp "$SRC" "$BUILDDIR/" || die "Failed to copy file $SRC to build dir: $BUILDDIR."
        done

        # Copy java code to chroot environment
        for SRC in ExaCompilationException.java       ExaExportSpecification.java      ExaIteratorImpl.java \
                            ExaCompiler.java                   ExaExportSpecificationImpl.java  ExaMetadata.java \
                            ExaConnectionAccessException.java  ExaImportSpecification.java      ExaMetadataImpl.java \
                            ExaConnectionInformation.java      ExaImportSpecificationImpl.java  ExaUndefinedSingleCallException.java \
                            ExaConnectionInformationImpl.java  ExaIterationException.java       ExaWrapper.java \
                            ExaDataTypeException.java          ExaIterator.java
        do
        cp "javacontainer/$SRC" "$BUILDDIR/java_src/com/exasol/" || die "Failed to copy file $SRC to $BUILDDIR/java_src/com/exasol/."
        done
    fi

    if [ "$ENABLE_JAVA_IMPL" = "yes" ]; then
        # create java wrapper from swig files
        mkdir -p java_src/com/exasol/swig || die "Failed to create directory java_src/com/exasol/swig"
        swig -O -DEXTERNAL_PROCESS -Wall -c++ -java -addextern -module exascript_java -package com.exasol.swig -outdir java_src/com/exasol/swig -o exascript_java_tmp.cc exascript.i || die "SWIG compilation failed."
        swig -DEXTERNAL_PROCESS -c++ -java -external-runtime exascript_java_tmp.h || die "SWIG compilation failed."

        python ./build_integrated.py exascript_java_int.h java_src/com/exasol/swig/exascript_java.java java_src/com/exasol/swig/exascript_javaJNI.java java_src/com/exasol/swig/SWIGVM_datatype_e.java java_src/com/exasol/swig/SWIGVM_itertype_e.java java_src/com/exasol/swig/Metadata.java java_src/com/exasol/swig/TableIterator.java java_src/com/exasol/swig/ResultHandler.java java_src/com/exasol/swig/ConnectionInformationWrapper.java java_src/com/exasol/swig/ImportSpecificationWrapper.java java_src/com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ImportSpecification.java java_src/com/exasol/swig/ExportSpecificationWrapper.java java_src/com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ExportSpecification.java					 

        python ./filter_swig_code.py exascript_java.cc exascript_java_tmp.cc
        python ./filter_swig_code.py exascript_java.h exascript_java_tmp.h

        LIBS="-ljvm $LIBS"
        #LDFLAGS="-L/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64 -L/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server -Wl,-rpath,/usr/lib/jvm/java-8-openjdk-amd64/lib/amd64 -Wl,-rpath,/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server $LDFLAGS"
        #CXXFLAGS="-DENABLE_JAVA_VM -I/usr/lib/jvm/java-8-openjdk-amd64/include -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux $CXXFLAGS"
        #CXXFLAGS_UNOPT="-DENABLE_JAVA_VM -I/usr/lib/jvm/java-8-openjdk-amd64/include -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux $CXXFLAGS_UNOPT"
        LDFLAGS="$JAVA_FLAGS $LDFLAGS"
        CXXFLAGS="-DENABLE_JAVA_VM $JAVA_FLAGS $CXXFLAGS"
        CXXFLAGS_UNOPT="-DENABLE_JAVA_VM $JAVA_FLAGS $CXXFLAGS_UNOPT"

        LDFLAGS="-L/usr/lib/jvm/java-9-openjdk-amd64/lib -L/usr/lib/jvm/java-9-openjdk-amd64/lib/amd64 -L/usr/lib/jvm/java-9-openjdk-amd64/lib/amd64/server -Wl,-rpath,/usr/lib/jvm/java-9-openjdk-amd64/lib -Wl,-rpath,/usr/lib/jvm/java-9-openjdk-amd64/lib/amd64 -Wl,-rpath,/usr/lib/jvm/java-9-openjdk-amd64/lib/amd64/server $LDFLAGS"
        
        CXXFLAGS="-DENABLE_JAVA_VM -I/usr/lib/jvm/java-9-openjdk-amd64/include -I/usr/lib/jvm/java-9-openjdk-amd64/include/linux $CXXFLAGS"
        CXXFLAGS_UNOPT="-DENABLE_JAVA_VM -I/usr/lib/jvm/java-9-openjdk-amd64/include -I/usr/lib/jvm/java-9-openjdk-amd64/include/linux $CXXFLAGS_UNOPT"



        g++ -o exascript_java.o -c exascript_java.cc $CXXFLAGS_UNOPT  || die "Failed to compile exascript_java.o"
        g++ -o javacontainer.o -c javacontainer.cc $CXXFLAGS || die "Failed to compile javacontainer.o"
        
        CONTAINER_CLIENT_OBJECT_FILES="exascript_java.o javacontainer.o $CONTAINER_CLIENT_OBJECT_FILES"

    # compile java code and create jar file for java container
    mkdir -p udf || die "Failed to create udf directory"
    #javac -bootclasspath "/usr/lib/jvm/java-9-openjdk-amd64/jrt-fs.jar" -source 1.8 -target 1.8 -classpath udf:java_src -d udf -sourcepath java_src \
    javac -classpath udf:java_src -d udf -sourcepath java_src \
        java_src/com/exasol/ExaCompilationException.java \
        java_src/com/exasol/ExaConnectionAccessException.java \
        java_src/com/exasol/ExaConnectionInformation.java \
        java_src/com/exasol/ExaDataTypeException.java \
        java_src/com/exasol/ExaImportSpecification.java \
        java_src/com/exasol/ExaExportSpecification.java \
        java_src/com/exasol/ExaIterationException.java \
        java_src/com/exasol/ExaIterator.java \
        java_src/com/exasol/ExaMetadata.java \
        java_src/com/exasol/ExaCompiler.java \
        java_src/com/exasol/ExaConnectionInformationImpl.java \
        java_src/com/exasol/ExaImportSpecificationImpl.java \
        java_src/com/exasol/ExaExportSpecificationImpl.java \
        java_src/com/exasol/ExaIteratorImpl.java \
        java_src/com/exasol/ExaMetadataImpl.java \
        java_src/com/exasol/ExaUndefinedSingleCallException.java \
        java_src/com/exasol/ExaWrapper.java \
        java_src/com/exasol/swig/ConnectionInformationWrapper.java \
        java_src/com/exasol/swig/exascript_java.java \
        java_src/com/exasol/swig/exascript_javaJNI.java \
        java_src/com/exasol/swig/ImportSpecificationWrapper.java \
        java_src/com/exasol/swig/ExportSpecificationWrapper.java \
        java_src/com/exasol/swig/Metadata.java \
        java_src/com/exasol/swig/ResultHandler.java \
        java_src/com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ImportSpecification.java \
        java_src/com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ExportSpecification.java \
        java_src/com/exasol/swig/SWIGVM_datatype_e.java \
        java_src/com/exasol/swig/SWIGVM_itertype_e.java \
        java_src/com/exasol/swig/TableIterator.java \
    || die "Failed to compile java API"
    jar -cf udf/exaudf-api-src.jar \
        -C java_src com/exasol/ExaCompilationException.java \
        -C java_src com/exasol/ExaConnectionAccessException.java \
        -C java_src com/exasol/ExaConnectionInformation.java \
        -C java_src com/exasol/ExaDataTypeException.java \
        -C java_src com/exasol/ExaImportSpecification.java \
        -C java_src com/exasol/ExaExportSpecification.java \
        -C java_src com/exasol/ExaIterationException.java \
        -C java_src com/exasol/ExaIterator.java \
        -C java_src com/exasol/ExaMetadata.java \
        LICENSE-exasol-script-api.txt \
    || die "Failed to create exaudf-api-src.jar"
    jar -cf udf/exaudf-api.jar \
        -C udf com/exasol/ExaCompilationException.class \
        -C udf com/exasol/ExaConnectionAccessException.class \
        -C udf com/exasol/ExaConnectionInformation.class \
        -C udf com/exasol/ExaConnectionInformation\$ConnectionType.class \
        -C udf com/exasol/ExaDataTypeException.class \
        -C udf com/exasol/ExaImportSpecification.class \
        -C udf com/exasol/ExaExportSpecification.class \
        -C udf com/exasol/ExaIterator.class \
        -C udf com/exasol/ExaIterationException.class \
        -C udf com/exasol/ExaMetadata.class \
        LICENSE-exasol-script-api.txt \
    || die "Failed to create exaudf-api.jar"
    jar -cf udf/exaudf.jar \
        -C udf com/exasol/ExaCompilationException.class \
        -C udf com/exasol/ExaConnectionAccessException.class \
        -C udf com/exasol/ExaConnectionInformation.class \
        -C udf com/exasol/ExaConnectionInformation\$ConnectionType.class \
        -C udf com/exasol/ExaDataTypeException.class \
        -C udf com/exasol/ExaImportSpecification.class \
        -C udf com/exasol/ExaExportSpecification.class \
        -C udf com/exasol/ExaIterator.class \
        -C udf com/exasol/ExaIterationException.class \
        -C udf com/exasol/ExaMetadata.class \
        -C udf com/exasol/ExaCompiler.class \
        -C udf com/exasol/ExaCompiler\$JavaSource.class \
        -C udf com/exasol/ExaConnectionInformationImpl.class \
        -C udf com/exasol/ExaImportSpecificationImpl.class \
        -C udf com/exasol/ExaExportSpecificationImpl.class \
        -C udf com/exasol/ExaIteratorImpl.class \
        -C udf com/exasol/ExaMetadataImpl.class \
        -C udf com/exasol/ExaMetadataImpl\$1.class \
        -C udf com/exasol/ExaMetadataImpl\$ColumnInfo.class \
        -C udf com/exasol/ExaUndefinedSingleCallException.class \
        -C udf com/exasol/ExaWrapper.class \
        -C udf com/exasol/swig/exascript_java.class \
        -C udf com/exasol/swig/TableIterator.class \
        -C udf com/exasol/swig/SWIGVM_datatype_e.class \
        -C udf com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ImportSpecification.class \
        -C udf com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ExportSpecification.class \
        -C udf com/exasol/swig/Metadata.class \
        -C udf com/exasol/swig/ResultHandler.class \
        -C udf com/exasol/swig/SWIGVM_itertype_e.class \
        -C udf com/exasol/swig/exascript_javaJNI.class \
        -C udf com/exasol/swig/ConnectionInformationWrapper.class \
        -C udf com/exasol/swig/ImportSpecificationWrapper.class \
        -C udf com/exasol/swig/ExportSpecificationWrapper.class \
    || die "Failed to create exaudf.jar"


    fi

}