package(default_visibility = ["//visibility:public"])
exports_files(["filter_swig_code.py", "build_integrated.py","LICENSE-exasol-script-api.txt", 
	       "create_binary_wrapper.sh", "create_binary_wrapper_valgrind.sh", "create_binary_wrapper_valgrind_massif.sh"])

load("//:variables.bzl", "VM_ENABLED_DEFINES")

config_setting(
    name = "benchmark",
    define_values = {"benchmark": "true"},
)

config_setting(
    name = "python",
    define_values = {"python": "true"},
)

config_setting(
    name = "java",
    define_values = {"java": "true"},
)

config_setting(
    name = "bash",
    define_values = {"bash": "true"},
)

config_setting(
    name = "fast_binary",
    define_values = {
        "binary_type": "fast_binary",
        },
)

config_setting(
    name = "slow_wrapper",
    define_values = {
        "binary_type": "slow_wrapper",
        },
)

config_setting(
    name = "valgrind_wrapper",
    define_values = {
        "wrapper_type": "valgrind_wrapper",
        },
)

config_setting(
    name = "valgrind_massif_wrapper",
    define_values = {
        "wrapper_type": "valgrind_massif_wrapper",
        },
)

config_setting(
    name = "stdout_to_bucketfs",
    define_values = {
        "wrapper_type": "stdout_to_bucketfs",
        },
)

cc_library(
    name = "debug_message_h",
    hdrs = [
        "debug_message.h"
    ],
    alwayslink = True,
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
    name = "exaudfclient_bin",
    srcs = ["exaudfclient.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON2_DEPS+
          ["//exaudflib:exaudflib-deps"],
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"],
)

cc_binary(
    name = "exaudfclient_py3_bin",
    srcs = ["exaudfclient.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS+
            ["//exaudflib:exaudflib-deps"],
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"]
)

cc_binary(
    name = "exaudfclient_py2_static",
    srcs = ["exaudfclient.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON2_DEPS+
          ["//exaudflib:exaudflib-deps"] + [":libexaudflib_complete.so"],
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"],
)

cc_binary(
    name = "exaudfclient_py3_static",
    srcs = ["exaudfclient.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS+
            ["//exaudflib:exaudflib-deps"] + [":libexaudflib_complete.so"],
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"],
)

# Workarround for the hardcoded paths in exaudfclient for libexaudflib_complete.so and python_ext_dataframe.cc
# - libexaudflib_complete.so and python_ext_dataframe.cc get dynamically loaded, therefore the exaudfclient needs to know their paths
# - Most flexible way to provides these paths would environment variables
# - The exasol database can't provide these paths, because they depend on the container
# - A workarround to provide these paths would be wrapper bash script which set these environment variables
# - For performance reason, we can not wrap the binary into a shell script, as such this is only for local debugging and testing
#   and we hardcode the paths for the production container

sh_library(
    name = "wrapper_generator_bin",
    srcs=select({
        "//:valgrind_wrapper": ["//:create_binary_wrapper_valgrind.sh"],
        "//:valgrind_massif_wrapper": ["//:create_binary_wrapper_valgrind_massif.sh"],
        "//:stdout_to_bucketfs": ["//:create_binary_wrapper_stdout_to_bucketfs.sh"],
        "//conditions:default": ["//:create_binary_wrapper.sh"]
    })
) 

SLOW_WRAPPER_BINARY_PY2="""$(location //:wrapper_generator_bin) "$(location exaudfclient_bin)" "$(location exaudfclient)" "$(location exaudfclient.template.sh)" """
FAST_BINARY_PY2="""cp "$(location exaudfclient_bin)" "$(location exaudfclient)" """
CREATE_BINARY_PY2_SCRIPT=select({
        "//:fast_binary": FAST_BINARY_PY2,
        "//:slow_wrapper": SLOW_WRAPPER_BINARY_PY2,
        "//conditions:default": FAST_BINARY_PY2
    })
genrule(
    name = "exaudfclient",
    cmd = CREATE_BINARY_PY2_SCRIPT,
    outs = ["exaudfclient"],
    srcs = [":exaudfclient_bin", "//:libexaudflib_complete.so", "exaudfclient.template.sh","//:wrapper_generator_bin"],
    output_to_bindir = True
)

SLOW_WRAPPER_BINARY_PY3="""$(location //:wrapper_generator_bin) "$(location exaudfclient_py3_bin)" "$(location exaudfclient_py3)" "$(location exaudfclient.template.sh)" """
FAST_BINARY_PY3="""cp "$(location exaudfclient_py3_bin)" "$(location exaudfclient_py3)" """
CREATE_BINARY_PY3_SCRIPT=select({
        "//:fast_binary": FAST_BINARY_PY3,
        "//:slow_wrapper": SLOW_WRAPPER_BINARY_PY3,
        "//conditions:default": FAST_BINARY_PY3
    })
genrule(
    name = "exaudfclient_py3",
    cmd = CREATE_BINARY_PY3_SCRIPT,
    outs = ["exaudfclient_py3"],
    srcs = [":exaudfclient_py3_bin", "//:libexaudflib_complete.so", "exaudfclient.template.sh", "//:wrapper_generator_bin"],
    output_to_bindir = True
)

