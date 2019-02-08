package(default_visibility = ["//visibility:public"])

genrule(
    name = "exascript_java_tmp_cc",
    cmd = """
        mkdir -p $(location java_src)/com/exasol/swig
        swig -v -O -DEXTERNAL_PROCESS -Wall -c++ -java -addextern -module exascript_java -package com.exasol.swig -outdir $(location java_src)/com/exasol/swig -o "$(location exascript_java_tmp.cc)" "$(location //:exascript.i)"
        """,
    outs = ["exascript_java_tmp.cc","java_src"],
    srcs = ["//:exascript.i","//:script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "exascript_java_tmp_h",
    cmd = """
        swig -v -DEXTERNAL_PROCESS -c++ -java -external-runtime "$(location exascript_java_tmp.h)"
        """,
    outs = ["exascript_java_tmp.h"],
    srcs = ["//:exascript.i","//:script_data_transfer_objects_wrapper.h", ":exascript_java_tmp_cc"]
)

genrule(
    name = "exascript_java_int",
    cmd = """
        cp -r $(locations :exascript_java_tmp_cc) . &&
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
            cp -r -L filter_swig_code.py $(locations :exascript_java_tmp_cc) $$TMPDIR && 
            (cd $$TMPDIR &&
            python ./filter_swig_code.py "exascript_java.cc" "exascript_java_tmp.cc") &&
            cp $$TMPDIR/exascript_java.cc "$@" &&
            rm -rf $$TMPDIR
            """,
    outs = ["exascript_java.cc"],
    srcs = [":exascript_java_tmp_cc"],
    tools = ["//:filter_swig_code.py"]
)
cc_library(
    name = "exascript_java",
    srcs = [":filter_swig_code_exascript_java_cc"],
    deps = ["@java_linux//:java_linux","//:exaudflib"]
)

cc_library(
    name = "javacontainer",
    srcs = [":javacontainer.cc"],
    hdrs = [":exascript_java_int", ":filter_swig_code_exascript_java_h","exascript_java_jni_decl.h"],
    include_prefix = ".",
    data = [":exaudf-api", ":exaudf"],
    deps = ["@ssl//:ssl","@java_linux//:java_linux","//:exaudflib","//:scriptoptionlines",":exascript_java"]
)

java_library(
    name = "exaudf",
    srcs = glob([":exascript_java_tmp_cc/java_src/com/exasol/swig/*.java",":Exa*.java"])
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

