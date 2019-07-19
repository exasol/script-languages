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
    name = "exaudfclient_wrong_rpath_py2",
    srcs = ["exaudfclient.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON2_DEPS,
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"]
)

cc_binary(
    name = "exaudfclient_wrong_rpath_py3",
    srcs = ["exaudfclient.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS,
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"]
)

# Workarounds for the problems of JNI/JVM with rpath's used by Bazel.
# Problem Description:
# - Bazel creates for all external local repositories symlinks into its build directory
# - Bazel than writes during compilation these paths to the rpath of the binary
# - JNI/JVM uses the rpath for loading additonal shared libraries, 
#   but for a unknown reason it seems to truncate these pathes. 
# - We assume two possible reason, the symlink paths of bazel contain a @, or
#   the paths are to long and get for that reason truncated
# - In the end, it we wasn't able to convinve JNI/JVM from other pathes 
#   for its libraries, as the original pathes where apt installed the files.
# - We developed two workarrounds:
#   1. Replace the paths for java in the rpath of the binary (//:exaudfclient_replace_rpath, //:exaudfclient_py3_replace_rpath)
#       - faster than the second workarround, because starting a script to setup the environment variables takes a moment
#       - with this workarround the paths for exaudflib-complete.so and python_ext_dataframe.so need to be hard coded into the binary,
#         whith the consequence that the binary only works in the UDF container in the database or in docker in a very specific directory
#       - Doesn't change the LD_LIBRARY_PATH environment variable
#       - 
#   2. Wrap the binary into a bash script which setups the LD_LIBRARY_PATH (used when libraries not found in the rpath) 
#      and environment variables for exaudflib-complete.so and python_ext_dataframe.so
#       - slower than the first workarround
#       - binaries work everywhere
#       - Java libraries directories are appended to LD_LIBRARY_PATH
#       - better for local testing and debugging

###########################################################################################
################################## Workarround 1 ##########################################
###########################################################################################
genrule(
    name = "exaudfclient_py2_replace_rpath",
    cmd = """
        $(location fix_rpath.sh) "$(location exaudfclient_wrong_rpath_py2)" "$(location exaudfclient)"
    """,
    outs = ["exaudfclient"],
    srcs = [":exaudfclient_wrong_rpath_py2", "fix_rpath.sh",":libexaudflib_complete.so"],
    output_to_bindir = True
)

genrule(
    name = "exaudfclient_py3_replace_rpath",
    cmd = """
        $(location fix_rpath.sh) "$(location exaudfclient_wrong_rpath_py3)" "$(location exaudfclient_py3)"
    """,
    outs = ["exaudfclient_py3"],
    srcs = [":exaudfclient_wrong_rpath_py3", "fix_rpath.sh",":libexaudflib_complete.so"],
    output_to_bindir = True
)

###########################################################################################
################################## Workarround 2 ##########################################
###########################################################################################
genrule(
    name = "exaudfclient_py2_bash_wrapper",
    cmd = """
        $(location create_binary_wrapper.sh) "$(location exaudfclient_wrong_rpath_py2)" "$(location exaudfclient_py2.sh)" "$(location exaudfclient.template.sh)"
    """,
    outs = ["exaudfclient_py2.sh"],
    srcs = [":exaudfclient_wrong_rpath_py2", "libexaudflib_complete.so", "exaudfclient.template.sh","create_binary_wrapper.sh"],
    output_to_bindir = True
)

genrule(
    name = "exaudfclient_py3_bash_wrapper",
    cmd = """
        $(location create_binary_wrapper.sh) "$(location exaudfclient_wrong_rpath_py3)" "$(location exaudfclient_py3.sh)" "$(location exaudfclient.template.sh)"
    """,
    outs = ["exaudfclient_py3.sh"],
    srcs = [":exaudfclient_wrong_rpath_py3", "libexaudflib_complete.so", "exaudfclient.template.sh", "create_binary_wrapper.sh"],
    output_to_bindir = True
)