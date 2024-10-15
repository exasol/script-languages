
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


VM_ENABLED_DEPS=select({
        "//:benchmark": ["//base/benchmark_container:benchmark_container"],
        "//conditions:default": []
    }) + select({
        "//:java": ["//base/javacontainer:javacontainer"],
        "//conditions:default": []
    }) + select({
        "//:bash": ["//base/streaming_container:streamingcontainer"],
        "//conditions:default": []
    })

VM_PYTHON3_DEPS=select({
        "//:python": ["//base/python/python3:pythoncontainer"],
        "//conditions:default": []
    })

cc_binary(
    name = "exaudfclient_bin",
    srcs = ["exaudfclient.cc", "//base:load_dynamic"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//base/exaudflib:header", "//base/utils:utils"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS+
           ["//base/exaudflib:exaudflib-deps"],
    defines = VM_ENABLED_DEFINES,
    data = ["//base:libexaudflib_complete.so"]
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
## We need to explicitly declare the dependency of protobuf/zmq here, as the exaudflib is a static lib (//base/exaudflib:exaudflib)
## and hence does not contain dependency information. We cannot declare the shared lib (:exaudflib_complete.so)
## as dependency as it is a binary for bazel.

cc_binary(
    name = "exaudfclient_static_bin",
    srcs = ["exaudfclient.cc", "//base:load_dynamic"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//base/exaudflib:header", "//base/utils:utils"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS+
            ["//base/exaudflib:exaudflib-deps"] +
            [ "@zmq//:zmq", "@protobuf//:protobuf"],
    defines = VM_ENABLED_DEFINES,
    data = ["//base:libexaudflib_complete.so"],
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
        "//:valgrind_wrapper": ["//base:create_binary_wrapper_valgrind.sh"],
        "//:valgrind_massif_wrapper": ["//base:create_binary_wrapper_valgrind_massif.sh"],
        "//:stdout_to_bucketfs": ["//base:create_binary_wrapper_stdout_to_bucketfs.sh"],
        "//conditions:default": ["//base:create_binary_wrapper.sh"]
    })
)

SLOW_WRAPPER_BINARY="""$(location //:wrapper_generator_bin) "$(location exaudfclient_bin)" "$(location exaudfclient)" "$(location //base:exaudfclient.template.sh)" """
FAST_BINARY="""cp "$(location exaudfclient_bin)" "$(location exaudfclient)" """
CREATE_BINARY_SCRIPT=select({
        "//:fast_binary": FAST_BINARY,
        "//:slow_wrapper": SLOW_WRAPPER_BINARY,
        "//conditions:default": FAST_BINARY
    })
genrule(
    name = "exaudfclient",
    cmd = CREATE_BINARY_SCRIPT,
    outs = ["exaudfclient"],
    srcs = [":exaudfclient_bin", "//base:libexaudflib_complete.so", "//base:exaudfclient.template.sh", "//:wrapper_generator_bin"],
    output_to_bindir = True
)

SLOW_WRAPPER_STATIC_BINARY="""$(location //:wrapper_generator_bin) "$(location exaudfclient_static_bin)" "$(location exaudfclient_static)" "$(location //base:exaudfclient.template.sh)" """
FAST_BINARY_STATIC="""cp "$(location exaudfclient_static_bin)" "$(location exaudfclient_static)" """
CREATE_STATIC_BINARY_SCRIPT=select({
        "//:fast_binary": FAST_BINARY_STATIC,
        "//:slow_wrapper": SLOW_WRAPPER_STATIC_BINARY,
        "//conditions:default": FAST_BINARY_STATIC
    })

genrule(
    name = "exaudfclient_static",
    cmd = CREATE_STATIC_BINARY_SCRIPT,
    outs = ["exaudfclient_static"],
    srcs = [":exaudfclient_static_bin", "//base:libexaudflib_complete.so", "//base:exaudfclient.template.sh", "//:wrapper_generator_bin"],
    output_to_bindir = True
)
