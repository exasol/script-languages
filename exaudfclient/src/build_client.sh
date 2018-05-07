#!/bin/sh
SRCDIR="$1"
BUILDDIR="$2"
OUTPUTDIR="$3"

if [ X"$SRCDIR" = X"" -o X"$OUTPUTDIR" = X"" -o X"$BUILDDIR" = X"" ]; then
    echo "Usage: $0 <srcdir> <outputdir> <builddir>" >&2
    exit 1
fi

die() { echo "ERROR:" "$@" >&2; exit 1; }

#set -x

cd $SRCDIR || die "Cannot change the current directory to /exasol_src"

#OUTPUTDIR="/exasol"
#BUILDDIR="/exasol_src/build"
mkdir -p  "$BUILDDIR"  "$BUILDDIR/java_src/com/exasol" || die "Failed to create directories in $DESTDIR"

# Generate swigcontainers_ext.h
cpp -DEXTERNAL_PROCESS swigcontainers.h | sed 's/^\$/#/; /^# *[0-9] */d; /^ *$/d' >"$BUILDDIR/swigcontainers_ext.h"

# Copy source code to the chroot environment
for SRC in \
        zmqcontainer.proto exascript.i filter_swig_code.py zmqcontainerclient.cc \
        rcontainer.cc exascript_r_preset.R exascript_r_wrap.R \
        pythoncontainer.cc exascript_python_preset.py exascript_python_wrap.py \
        javacontainer.cc exascript_java_jni_decl.h scriptoptionlines.h scriptoptionlines.cc \
	    script_data_transfer_objects* LICENSE-exasol-script-api.txt
do
    cp "$SRC" "$BUILDDIR/" || die "Failed to copy file $SRC to chroot environment."
done

# Copy java code to chroot environment
for SRC in ExaCompilationException.java       ExaExportSpecification.java      ExaIteratorImpl.java \
           ExaCompiler.java                   ExaExportSpecificationImpl.java  ExaMetadata.java \
           ExaConnectionAccessException.java  ExaImportSpecification.java      ExaMetadataImpl.java \
           ExaConnectionInformation.java      ExaImportSpecificationImpl.java  ExaUndefinedSingleCallException.java \
           ExaConnectionInformationImpl.java  ExaIterationException.java       ExaWrapper.java \
           ExaDataTypeException.java          ExaIterator.java
do
    cp "javacontainer/$SRC" "$BUILDDIR/java_src/com/exasol/" || die "Failed to copy file $SRC to chroot environment."
done

cd $BUILDDIR || die "No $BUILDDIR directory found"

# create source code from proto files
protoc -I. zmqcontainer.proto --cpp_out=. || die "Failed to create C++ proto files."
protoc zmqcontainer.proto --python_out=. || die "Failed to create Python proto files."

# create python wrapper from swig files
swig -O -DEXTERNAL_PROCESS -Wall -c++ -python -addextern -module exascript_python -o exascript_python_tmp.cc exascript.i || die "SWIG compilation failed."
swig -DEXTERNAL_PROCESS -c++ -python -external-runtime exascript_python_tmp.h || die "SWIG compilation failed."

# create R wrapper from swig files
swig -O -DEXTERNAL_PROCESS -Wall -c++ -r -addextern -module exascript_r -o exascript_r_tmp.cc exascript.i >/dev/null 2>&1 || die "SWIG compilation failed."
swig -DEXTERNAL_PROCESS -c++ -r -external-runtime exascript_r_tmp.h || die "SWIG compilation failed."

# create java wrapper from swig files
mkdir -p java_src/com/exasol/swig || die "Failed to create directory java_src/com/exasol/swig"
swig -O -DEXTERNAL_PROCESS -Wall -c++ -java -addextern -module exascript_java -package com.exasol.swig -outdir java_src/com/exasol/swig -o exascript_java_tmp.cc exascript.i || die "SWIG compilation failed."
swig -DEXTERNAL_PROCESS -c++ -java -external-runtime exascript_java_tmp.h || die "SWIG compilation failed."

# cleanup and finalize generated swig code
python ./filter_swig_code.py || die "Failed to filter SWIG code."

# compile zmqcontainerclient
export CXXFLAGS="-DRT_ROOT_DIRECTORY=\"\\\"/usr\\\"\" -DENABLE_PYTHON_VM -DENABLE_R_VM -DENABLE_JAVA_VM  -I. -I/usr/include/python2.7 -I/usr/share/R/include -I/usr/lib/jvm/java-8-openjdk-amd64/include -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux -Wall -Werror -fPIC -pthread -DNDEBUG -std=c++14"

export CXXFLAGS2="-O3 $CXXFLAGS"

export LDFLAGS="-L/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64 -L/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server -Wl,-rpath,/usr/lib/jvm/java-8-openjdk-amd64/lib/amd64 -Wl,-rpath,/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server"
g++ -o zmqcontainerclient.o -c zmqcontainerclient.cc $CXXFLAGS2 || die "Failed to compile zmqcontainerclient.o"
g++ -o exascript_python.o -c exascript_python.cc $CXXFLAGS2 || die "Failed to compile exascript_python.o"
g++ -o pythoncontainer.o -c pythoncontainer.cc $CXXFLAGS2 || die "Failed to compile pythoncontainer.o"
g++ -o exascript_r.o -c exascript_r.cc $CXXFLAGS2 || die "Failed to compile exascript_r.o"
g++ -o rcontainer.o -c rcontainer.cc $CXXFLAGS2 || die "Failed to compile rcontainer.o"
g++ -o exascript_java.o -c exascript_java.cc $CXXFLAGS || die "Failed to compile exascript_java.o"
g++ -o javacontainer.o -c javacontainer.cc $CXXFLAGS2 || die "Failed to compile javacontainer.o"
g++ -o zmqcontainer.pb.o -c zmqcontainer.pb.cc $CXXFLAGS2 || die "Failed to compile zmqcontainer.pb.o"
g++ -o scriptDTOWrapper.o -c script_data_transfer_objects_wrapper.cc $CXXFLAGS2 || die "Failed to compile scriptDTOWrapper.o"
g++ -o scriptDTO.o -c script_data_transfer_objects.cc $CXXFLAGS2 || die "Failed to compile scriptDTO.o"
g++ -o scriptoptionlines.o -c scriptoptionlines.cc $CXXFLAGS2 || die "Failed to compile scriptoptionlines.o"
g++ -o zmqcontainerclient zmqcontainerclient.o exascript_python.o pythoncontainer.o exascript_r.o rcontainer.o exascript_java.o javacontainer.o zmqcontainer.pb.o scriptDTOWrapper.o scriptDTO.o scriptoptionlines.o -lpython2.7 -lR -ljvm -lzmq -lprotobuf -lpthread -lcrypto $LDFLAGS || die "Failed to compile zmqcontainerclient"

# compile java code and create jar file for java container
mkdir -p udf || die "Failed to create udf directory"
javac -bootclasspath "/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/rt.jar" -source 1.7 -target 1.7 -classpath udf:java_src -d udf -sourcepath java_src \
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



# Create output files
cp -a "$BUILDDIR/zmqcontainerclient" "$OUTPUTDIR/exaudfclient" || die "Failed to create $OUTPUTDIR/exaudfclient"
chmod +x "$OUTPUTDIR/exaudfclient" || die "Failed chmod of $OUTPUTDIR/exaudfclient"
cp -a "$BUILDDIR/udf/exaudf-api-src.jar" "$OUTPUTDIR/exasol-script-api-sources.jar" || die "Failed to create $OUTPUTDIR/exasol-script-api-sources.jar"
cp -a "$BUILDDIR/udf/exaudf-api.jar" "$OUTPUTDIR/exasol-script-api.jar" || die "Failed to create $OUTPUTDIR/exasol-script-api.jar"
cp -a "$BUILDDIR/udf/exaudf.jar" "$OUTPUTDIR/exaudf.jar" || die "Failed to create $OUTPUTDIR/exaudf.jar"

