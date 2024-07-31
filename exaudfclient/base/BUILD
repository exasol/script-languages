package(default_visibility = ["//visibility:public"])
exports_files(["filter_swig_code.py", "build_integrated.py",
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
)

# Using cc_binary to build a shared library as output target is a workaround, because
# Bazel's cc_library are only intermediate stages and can be used as output target.
# It is necessary to include //exaudflib:exaudflib into deps and srcs, because
# otherwise dlmopen won't find the symbols of exaudflib and its dependencies. The target //exaudflib:exaudflib-deps
# won't work either, because it only contains the libraries as dependencies.
# Bazel builts a static library (.a) and a dynamic library (.so) out of exaudflib:exaudflib.
# In the libexaudflib_complete.so-2.params, we saw that Bazel uses both libraries to built libexaudflib_complete.so.
# Experiments have shown that we get missing symbols during loading libexaudflib_complete.so, if we only built the static library. 
cc_binary(
    name = "libexaudflib_complete.so",
    linkshared = 1,
    srcs = ["//exaudflib:exaudflib"],
    deps = ["//exaudflib:exaudflib"],
    defines = VM_ENABLED_DEFINES,
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

VM_PYTHON3_DEPS=select({
        ":python": ["//python/python3:pythoncontainer"],
        "//conditions:default": []
    })

cc_binary(
    name = "exaudfclient_py3_bin",
    srcs = ["exaudfclient.cc", "load_dynamic.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS+
           ["//exaudflib:exaudflib-deps"],
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"]
)

## The purpose of the static binaries is to verify if the linker namespace test work correctly.
## It is crucial that all linker symbols of exaudflib and all it's dependency are loaded into a new linker namespace,
## so that the user can load it's own versions of those dependencies without any conflict.
## With this binary we simulate an error in our build system, that is a direct depedency to protobuf/zmq,
## which then must be detected with the linker namespace tests:
## test/linker_namespace_sanity/linker_namespace_sanity.py checks the wrong configuration
## Besides this the test under test/python3/all/linker_namespace.py checks the normal build, which expects
## not to find any occurence of the dependencies (protobuf/zmq) in the primary linker namespace.
##
## We need to explicitly declare the dependency of protobuf/zmq here, as the exaudflib is a static lib (//exaudflib:exaudflib)
## and hence does not contain dependency information. We cannot declare the shared lib (:exaudflib_complete.so)
## as dependency as it is a binary for bazel.

cc_binary(
    name = "exaudfclient_py3_static_bin",
    srcs = ["exaudfclient.cc", "load_dynamic.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS+
            ["//exaudflib:exaudflib-deps"] + [ "@libzmq//:zmq", "@protobuf//:protobuf"],
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

SLOW_WRAPPER_STATIC_BINARY_PY3="""$(location //:wrapper_generator_bin) "$(location exaudfclient_py3_static_bin)" "$(location exaudfclient_py3_static)" "$(location exaudfclient.template.sh)" """
FAST_BINARY_STATIC_PY3="""cp "$(location exaudfclient_py3_static_bin)" "$(location exaudfclient_py3_static)" """
CREATE_STATIC_BINARY_PY3_SCRIPT=select({
        "//:fast_binary": FAST_BINARY_STATIC_PY3,
        "//:slow_wrapper": SLOW_WRAPPER_STATIC_BINARY_PY3,
        "//conditions:default": FAST_BINARY_STATIC_PY3
    })

genrule(
    name = "exaudfclient_py3_static",
    cmd = CREATE_STATIC_BINARY_PY3_SCRIPT,
    outs = ["exaudfclient_py3_static"],
    srcs = [":exaudfclient_py3_static_bin", "//:libexaudflib_complete.so", "exaudfclient.template.sh", "//:wrapper_generator_bin"],
    output_to_bindir = True
)
