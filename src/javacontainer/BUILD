package(default_visibility = ["//visibility:public"])

genrule(
    name = "exascript_java_tmp_cc",
    cmd = """
        mkdir -p $(location java_src)/com/exasol/swig
        swig -v -O -DEXTERNAL_PROCESS -Wall -c++ -java -addextern -module exascript_java -package com.exasol.swig -outdir $(location java_src)/com/exasol/swig -o "$(location exascript_java_tmp.cc)" "$(location //exaudflib:exascript.i)"
        """,
    outs = ["exascript_java_tmp.cc","java_src"],
    srcs = ["//exaudflib:exascript.i","//exaudflib:script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "exascript_java_tmp_h",
    cmd = """
        swig -v -DEXTERNAL_PROCESS -c++ -java -external-runtime "$(location exascript_java_tmp.h)"
        """,
    outs = ["exascript_java_tmp.h"],
    srcs = ["//exaudflib:exascript.i","//exaudflib:script_data_transfer_objects_wrapper.h", ":exascript_java_tmp_cc"]
)

genrule(
    name = "exascript_java_int",
    cmd = """
        cp -r $(locations :exascript_java_tmp_cc) .
        FILES=`ls java_src/com/exasol/swig/*.java`
        python build_integrated.py "$(location exascript_java_int.h)" $$FILES
    """,
    outs = ["exascript_java_int.h"],
    srcs = [":exascript_java_tmp_cc"],
    tools = ["//:build_integrated.py"]
)

genrule(
    name = "filter_swig_code_exascript_java_h",
    cmd = 'python ./filter_swig_code.py "$@" "$<"',
    outs = ["exascript_java.h"],
    srcs = [":exascript_java_tmp_h"],
    tools = ["//:filter_swig_code.py"]
)

genrule(
    name = "filter_swig_code_exascript_java_cc",
    cmd = """
            TMPDIR=`mktemp -d`
            cp -r -L filter_swig_code.py $(locations :exascript_java_tmp_cc) $$TMPDIR
            (cd $$TMPDIR &&
            python ./filter_swig_code.py "exascript_java.cc" "exascript_java_tmp.cc")
            cp $$TMPDIR/exascript_java.cc "$(location exascript_java.cc)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/ConnectionInformationWrapper.java "$(location ConnectionInformationWrapper.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/exascript_java.java "$(location exascript_java.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/exascript_javaJNI.java "$(location exascript_javaJNI.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/ExportSpecificationWrapper.java "$(location ExportSpecificationWrapper.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/ImportSpecificationWrapper.java "$(location ImportSpecificationWrapper.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/Metadata.java "$(location Metadata.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/ResultHandler.java "$(location ResultHandler.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ExportSpecification.java "$(location SWIGTYPE_p_ExecutionGraph__ExportSpecification.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ImportSpecification.java "$(location SWIGTYPE_p_ExecutionGraph__ImportSpecification.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/SWIGVM_datatype_e.java "$(location SWIGVM_datatype_e.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/SWIGVM_itertype_e.java "$(location SWIGVM_itertype_e.java)"
            cp -r $$TMPDIR/java_src/com/exasol/swig/TableIterator.java "$(location TableIterator.java)"
            rm -rf $$TMPDIR
            """,
    outs = [
            "exascript_java.cc",
            "ConnectionInformationWrapper.java",
            "exascript_java.java",
            "exascript_javaJNI.java",
            "ExportSpecificationWrapper.java",
            "ImportSpecificationWrapper.java",
            "Metadata.java",
            "ResultHandler.java",
            "SWIGTYPE_p_ExecutionGraph__ExportSpecification.java",
            "SWIGTYPE_p_ExecutionGraph__ImportSpecification.java",
            "SWIGVM_datatype_e.java",
            "SWIGVM_itertype_e.java",
            "TableIterator.java",
            ],
    srcs = [":exascript_java_tmp_cc"],
    tools = ["//:filter_swig_code.py"]
)

cc_library(
    name = "exascript_java",
    srcs = [":filter_swig_code_exascript_java_cc",],
    deps = ["@java//:java","//exaudflib:exaudflib-deps","//exaudflib:header"],
    copts= ["-O0","-fno-lto"],
    # We limit this target to -O0 (no optimization) and -fno-lto. because otherwise we get compiler warnings of the sort 
    # note: "code may be misoptimized unless -fno-strict-aliasing is used." 
    # Normally, this indicates that compiling with higher optimizations levels may break your progrom. Unitl now, we don't have enough tests
    # to verify if higher levels of optimizations might work. We needed to deactivate link time optimization, too, because this delay the compilation
    # until the exaudfclient binary is build and causes than there the same problem.
)

cc_library(
    name = "javacontainer",
    srcs = [":javacontainer.cc", ":dummy"],
    hdrs = [":exascript_java_int", ":filter_swig_code_exascript_java_h","exascript_java_jni_decl.h"],
    deps = ["@ssl//:ssl","@java//:java", ":exascript_java", "//exaudflib:header", "//:debug_message_h"],
)

#workaround to build jars together with javacontainer c++ library
genrule(
    name = "dummy",
    cmd = 'touch "$@"',
    outs = ["dummy.h"],
    srcs = [":exaudf-api", ":exaudf", "libexaudf-src.jar", "libexaudf-api-src.jar"],
)

java_library(
    name = "exaudf",
    srcs = glob(["*.java"])+
                [
                "ConnectionInformationWrapper.java",
                "exascript_java.java",
                "exascript_javaJNI.java",
                "ExportSpecificationWrapper.java",
                "ImportSpecificationWrapper.java",
                "Metadata.java",
                "ResultHandler.java",
                "SWIGTYPE_p_ExecutionGraph__ExportSpecification.java",
                "SWIGTYPE_p_ExecutionGraph__ImportSpecification.java",
                "SWIGVM_datatype_e.java",
                "SWIGVM_itertype_e.java",
                "TableIterator.java"
                ]
)

java_library(
    name = "exaudf-api",
    srcs = [
            ":ExaCompilationException.java",
            ":ExaConnectionAccessException.java",
            ":ExaConnectionInformation.java",
            ":ExaDataTypeException.java",
            ":ExaImportSpecification.java",
            ":ExaExportSpecification.java",
            ":ExaIterationException.java",
            ":ExaIterator.java",
            ":ExaMetadata.java"
            ],
    resources = ["//:LICENSE-exasol-script-api.txt"],
    deps = [":exaudf"]
)

