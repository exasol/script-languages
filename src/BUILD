package(default_visibility = ["//visibility:public"])
exports_files(["filter_swig_code.py", "build_integrated.py","LICENSE-exasol-script-api.txt"])

config_setting(
    name = "benchmark",
    define_values = {"benchmark": "true"},
)

config_setting(
    name = "python",
    define_values = {"python": "true"},
)

config_setting(
    name = "r",
    define_values = {"r": "true"},
)

config_setting(
    name = "java",
    define_values = {"java": "true"},
)

config_setting(
    name = "bash",
    define_values = {"bash": "true"},
)

load("//:variables.bzl", "VM_ENABLED_DEFINES")

cc_library(
    name = "debug_message_h",
    hdrs = [
        "debug_message.h"
    ],
)

# Using cc_binary to build a shared library as output target is a workaround, because
# Bazel's cc_library are only intermediate stage and can be used as output target.
# It is necessary to include all dependencies of //exaudflib:exaudflib into deps and srcs,
# otherwise dlmopen won't find the symbols from these dependencies. The target //exaudflib:exaudflib-deps
# won't work either, because it only contains the libraries only as dependencies.
cc_binary(
    name = "libexaudflib_complete.so",
    linkshared = 1,
    srcs = ["//exaudflib:exaudflib", 
            "//exaudflib:zmqcontainer", 
            "//exaudflib:script_data_transfer_objects_wrapper", 
            "//exaudflib:script_data_transfer_objects"
            ],
    deps = [ "@zmq//:zmqlib", "@protobuf//:protobuf"],
    defines = VM_ENABLED_DEFINES
)

VM_ENABLED_DEPS=select({
        ":benchmark": ["//benchmark_container:benchmark_container"],
        "//conditions:default": []
    }) + select({
        ":r": ["//rcontainer:rcontainer"],
        "//conditions:default": []
    }) + select({
        ":java": ["//javacontainer:javacontainer"],
        "//conditions:default": []
    }) + select({
        ":bash": ["//streaming_container:streamingcontainer"],
        "//conditions:default": []
    })

VM_PYTHON2_DEPS=select({
        ":python": ["//python/python2:pythoncontainer"],
        "//conditions:default": []
    }) 

VM_PYTHON3_DEPS=select({
        ":python": ["//python/python3:pythoncontainer"],
        "//conditions:default": []
    }) 

cc_binary(
    name = "exaudfclient",
    srcs = ["exaudfclient.cc"]+
            ["//exaudflib:scriptoptionlines"], # needed to avoid compiler/linker errors with -flto with languages with optimize
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON2_DEPS+
            ["//exaudflib:scriptoptionlines"]+ # needed to avoid compiler/linker errors with -flto with languages with optimize
            ["//exaudflib:exaudflib-deps"], # needed for compiling without language
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"]
)

#TODO 172.17.0.2:60292> exaudf/exaudfclient: error while loading shared libraries: libscriptoptionlines.so: cannot open shared object file: No such file or directory


cc_binary(
    name = "exaudfclient_py3",
    srcs = ["exaudfclient.cc"]+
            ["//exaudflib:scriptoptionlines"], # needed to avoid compiler/linker errors with -flto with languages with optimize          
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS+
            ["//exaudflib:scriptoptionlines"]+ # needed to avoid compiler/linker errors with -flto with languages with optimize
            ["//exaudflib:exaudflib-deps"], # needed for compiling without language
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"]
)

# Workarround for the hardcoded paths in exaudfclient for libexaudflib_complete.so and python_ext_dataframe.cc
# - libexaudflib_complete.so and python_ext_dataframe.cc get dynamically loaded, therefore the exaudfclient needs to know their paths
# - Most flexible way to provides these paths would environment variables
# - The exasol database can't provide these paths, because they depend on the container
# - A workarround to provide these paths would be wrapper bash script which set these environment variables
# - For performance reason, we can not wrap the binary into a shell script, as such this is only for local debugging and testing
#   and we hardcode the paths for the production container
genrule(
    name = "exaudfclient_py2_bash_wrapper",
    cmd = """
        $(location create_binary_wrapper.sh) "$(location exaudfclient)" "$(location exaudfclient.sh)" "$(location exaudfclient.template.sh)"
    """,
    outs = ["exaudfclient.sh"],
    srcs = [":exaudfclient", "libexaudflib_complete.so", "exaudfclient.template.sh","create_binary_wrapper.sh"],
    output_to_bindir = True
)

genrule(
    name = "exaudfclient_py3_bash_wrapper",
    cmd = """
        $(location create_binary_wrapper.sh) "$(location exaudfclient_py3)" "$(location exaudfclient_py3.sh)" "$(location exaudfclient.template.sh)"
    """,
    outs = ["exaudfclient_py3.sh"],
    srcs = [":exaudfclient_py3", "libexaudflib_complete.so", "exaudfclient.template.sh", "create_binary_wrapper.sh"],
    output_to_bindir = True
)