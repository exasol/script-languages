package(default_visibility = ["//visibility:public"])

genrule(
    name = "exascript_java_tmp_cc",
    cmd = """
        mkdir -p java_src/com/exasol/swig
        mkdir -p build_exascript_java_tmp_cc/exaudflib
        cp "$(location //base/exaudflib:swig/script_data_transfer_objects_wrapper.h)" "$(location //base/exaudflib:exascript.i)" build_exascript_java_tmp_cc/exaudflib
        cd build_exascript_java_tmp_cc
        swig -v -O -DEXTERNAL_PROCESS -Wall -c++ -java -addextern -module exascript_java -package com.exasol.swig -outdir "../java_src/com/exasol/swig" -o "../exascript_java_tmp.cc" exaudflib/exascript.i
        cd ..
        cp java_src/com/exasol/swig/ConnectionInformationWrapper.java $(location ConnectionInformationWrapper.java)
        cp java_src/com/exasol/swig/exascript_java.java $(location exascript_java.java)
        cp java_src/com/exasol/swig/exascript_javaJNI.java $(location raw/exascript_javaJNI.java)
        cp java_src/com/exasol/swig/ExportSpecificationWrapper.java $(location ExportSpecificationWrapper.java)
        cp java_src/com/exasol/swig/ImportSpecificationWrapper.java $(location ImportSpecificationWrapper.java)
        cp java_src/com/exasol/swig/Metadata.java $(location Metadata.java)
        cp java_src/com/exasol/swig/ResultHandler.java $(location raw/ResultHandler.java)
        cp java_src/com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ExportSpecification.java $(location SWIGTYPE_p_ExecutionGraph__ExportSpecification.java)
        cp java_src/com/exasol/swig/SWIGTYPE_p_ExecutionGraph__ImportSpecification.java $(location SWIGTYPE_p_ExecutionGraph__ImportSpecification.java)
        cp java_src/com/exasol/swig/SWIGVM_datatype_e.java $(location SWIGVM_datatype_e.java)
        cp java_src/com/exasol/swig/SWIGVM_itertype_e.java $(location SWIGVM_itertype_e.java)
        cp java_src/com/exasol/swig/TableIterator.java $(location raw/TableIterator.java)
        cp exascript_java_tmp.cc $(location exascript_java_tmp.cc)
    """,
    outs = ["ConnectionInformationWrapper.java",
            "exascript_java.java",
            "ExportSpecificationWrapper.java",
            "ImportSpecificationWrapper.java",
            "Metadata.java",
            "SWIGTYPE_p_ExecutionGraph__ExportSpecification.java",
            "SWIGTYPE_p_ExecutionGraph__ImportSpecification.java",
            "SWIGVM_datatype_e.java",
            "SWIGVM_itertype_e.java",
            "raw/exascript_javaJNI.java",
            "raw/ResultHandler.java",
            "raw/TableIterator.java",
            "exascript_java_tmp.cc"
    ],
    srcs = ["//base/exaudflib:exascript.i","//base/exaudflib:swig/script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "exascript_java_tmp_h",
    cmd = """
        mkdir build_exascript_java_tmp_h
        cp "$(location //base/exaudflib:swig/script_data_transfer_objects_wrapper.h)" "$(location //base/exaudflib:exascript.i)" build_exascript_java_tmp_h
        cp -r "$(location exascript_java_tmp.cc)" build_exascript_java_tmp_h
        cd build_exascript_java_tmp_h
        swig -v -DEXTERNAL_PROCESS -c++ -java -external-runtime "../$(location exascript_java_tmp.h)"
        """,
    outs = ["exascript_java_tmp.h"],
    srcs = ["//base/exaudflib:exascript.i","//base/exaudflib:swig/script_data_transfer_objects_wrapper.h", ":exascript_java_tmp.cc"]
)


genrule(
    name = "filter_swig_code_exascript_java_h",
    cmd = 'python3 $(location //base:filter_swig_code.py) "$@" "$<"',
    outs = ["exascript_java.h"],
    srcs = [":exascript_java_tmp_h"],
    tools = ["//base:filter_swig_code.py"]
)

genrule(
    name = "filter_swig_code_exascript_java_cc",
    cmd = """
            TMPDIR=`mktemp -d`
            mkdir -p "$$TMPDIR"/java_src/com/exasol/swig
            cp  $(location :raw/TableIterator.java) "$$TMPDIR"/java_src/com/exasol/swig/
            cp  $(location :raw/ResultHandler.java) "$$TMPDIR"/java_src/com/exasol/swig/
            cp  $(location :raw/exascript_javaJNI.java) "$$TMPDIR"/java_src/com/exasol/swig/
            find "$$TMPDIR"/java_src/com/exasol/swig/ -name *.java -type f -exec chmod 644 {} \\;
            cp -r -L $(location //base:filter_swig_code.py) $(location :exascript_java_tmp.cc) "$$TMPDIR"
            (cd "$$TMPDIR"
            python3 filter_swig_code.py "exascript_java.cc" "exascript_java_tmp.cc")
            cp "$$TMPDIR"/exascript_java.cc $(location :exascript_java.cc)
            cp "$$TMPDIR"/java_src/com/exasol/swig/TableIterator.java $(location TableIterator.java)
            cp "$$TMPDIR"/java_src/com/exasol/swig/ResultHandler.java $(location ResultHandler.java)
            cp "$$TMPDIR"/java_src/com/exasol/swig/exascript_javaJNI.java $(location exascript_javaJNI.java)
            rm -rf "$$TMPDIR"
            """,
    outs = [
            "exascript_java.cc",
            "TableIterator.java",
            "ResultHandler.java",
            "exascript_javaJNI.java"
            ],
    srcs = [":exascript_java_tmp.cc", ":raw/TableIterator.java",
            ":raw/ResultHandler.java", ":raw/exascript_javaJNI.java"],
    tools = ["//base:filter_swig_code.py"]
)

cc_library(
    name = "exascript_java",
    srcs = [":exascript_java.cc",],
    deps = ["@java//:java","//base/exaudflib:exaudflib-deps","//base/exaudflib:header"],
    copts= ["-O0","-fno-lto"],
    # We limit this target to -O0 (no optimization) and -fno-lto. because otherwise we get compiler warnings of the sort 
    # note: "code may be misoptimized unless -fno-strict-aliasing is used." 
    # Normally, this indicates that compiling with higher optimizations levels may break your progrom. Unitl now, we don't have enough tests
    # to verify if higher levels of optimizations might work. We needed to deactivate link time optimization, too, because this delay the compilation
    # until the exaudfclient binary is build and causes than there the same problem.
    alwayslink=True,
)

cc_library(
    name = "javacontainer",
    srcs = [":javacontainer.cc", ":javacontainer.h", ":javacontainer_impl.cc", ":javacontainer_impl.h",
            ":javacontainer_builder.h", ":javacontainer_builder.cc", ":dummy"],
    hdrs = [":filter_swig_code_exascript_java_h", "exascript_java_jni_decl.h"],
    deps = ["@java//:java", ":exascript_java", "//base/exaudflib:header",
            "//base/utils:utils","//base/javacontainer/script_options:java_script_option_lines",
            "//base/swig_factory:swig_factory_if"],
#    copts= ["-O0","-fno-lto"],
    alwayslink=True,
)

#workaround to build jars together with javacontainer c++ library
genrule(
    name = "dummy",
    cmd = 'touch "$@"',
    outs = ["dummy.h"],
    srcs = [":exaudf_deploy.jar"],
)

java_binary(
    name = "exaudf",
    srcs = glob(["*.java"])+
                #The following are generated classes
                [
                ":ConnectionInformationWrapper.java",
                ":exascript_java.java",
                ":exascript_javaJNI.java",
                ":ExportSpecificationWrapper.java",
                ":ImportSpecificationWrapper.java",
                ":Metadata.java",
                ":ResultHandler.java",
                ":SWIGTYPE_p_ExecutionGraph__ExportSpecification.java",
                ":SWIGTYPE_p_ExecutionGraph__ImportSpecification.java",
                ":SWIGVM_datatype_e.java",
                ":SWIGVM_itertype_e.java",
                ":TableIterator.java"
                ],
    deps = ["@maven//:com_exasol_udf_api_java"],
    create_executable=False,
)

exports_files([
    "ExaStackTraceCleaner.java",
    ])
