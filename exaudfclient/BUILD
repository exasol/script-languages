
load("//:variables.bzl", "VM_ENABLED_DEFINES")



VM_ENABLED_DEPS=select({
        "@exaudfclient_base//:benchmark": ["@exaudfclient_base//benchmark_container:benchmark_container"],
        "//conditions:default": []
    }) + select({
        "@exaudfclient_base//:java": ["@exaudfclient_base//javacontainer:javacontainer"],
        "//conditions:default": []
    }) + select({
        "@exaudfclient_base//:bash": ["@exaudfclient_base//streaming_container:streamingcontainer"],
        "//conditions:default": []
    })

VM_PYTHON3_DEPS=select({
        "@exaudfclient_base//:python": ["@exaudfclient_base//python/python3:pythoncontainer"],
        "//conditions:default": []
    })


genrule(
    name = "exaudflib_output_path_header",
    srcs = ["@exaudfclient_base//:libexaudflib_complete.so"],
    outs = ["exaudf_lib_output_path.h"],
    cmd = '''
      echo "#define CUSTOM_LIBEXAUDFLIB_PATH \\"/exaudf/external/$(rlocationpath @exaudfclient_base//:libexaudflib_complete.so)\\"" > $@
    ''',
)

cc_binary(
    name = "exaudfclient_bin",
    srcs = ["exaudfclient.cc", "@exaudfclient_base//:load_dynamic", "//:exaudflib_output_path_header"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["@exaudfclient_base//exaudflib:header", "@exaudfclient_base//utils:utils"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS+
           ["@exaudfclient_base//exaudflib:exaudflib-deps"],
    defines = VM_ENABLED_DEFINES,
    data = ["@exaudfclient_base//:libexaudflib_complete.so"]
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
    name = "exaudfclient_static_bin",
    srcs = ["exaudfclient.cc", "@exaudfclient_base//:load_dynamic", "//:exaudflib_output_path_header"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["@exaudfclient_base//exaudflib:header", "@exaudfclient_base//utils:utils"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS+
            ["@exaudfclient_base//exaudflib:exaudflib-deps"] +
            [ "@zmq//:zmq", "@protobuf//:protobuf"],
    defines = VM_ENABLED_DEFINES,
    data = ["@exaudfclient_base//:libexaudflib_complete.so"],
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
        "@exaudfclient_base//:valgrind_wrapper": ["@exaudfclient_base//:create_binary_wrapper_valgrind.sh"],
        "@exaudfclient_base//:valgrind_massif_wrapper": ["@exaudfclient_base//:create_binary_wrapper_valgrind_massif.sh"],
        "@exaudfclient_base//:stdout_to_bucketfs": ["@exaudfclient_base//:create_binary_wrapper_stdout_to_bucketfs.sh"],
        "//conditions:default": ["//base:create_binary_wrapper.sh"]
    })
)

SLOW_WRAPPER_BINARY="""$(location //:wrapper_generator_bin) "$(location exaudfclient_bin)" "$(location exaudfclient)" "$(location @exaudfclient_base//:exaudfclient.template.sh)" """
FAST_BINARY="""cp "$(location exaudfclient_bin)" "$(location exaudfclient)" """
CREATE_BINARY_SCRIPT=select({
        "@exaudfclient_base//:fast_binary": FAST_BINARY,
        "@exaudfclient_base//:slow_wrapper": SLOW_WRAPPER_BINARY,
        "//conditions:default": FAST_BINARY
    })
genrule(
    name = "exaudfclient",
    cmd = CREATE_BINARY_SCRIPT,
    outs = ["exaudfclient"],
    srcs = [":exaudfclient_bin", "@exaudfclient_base//:libexaudflib_complete.so", "@exaudfclient_base//:exaudfclient.template.sh", "//:wrapper_generator_bin"],
    output_to_bindir = True
)

SLOW_WRAPPER_STATIC_BINARY="""$(location //:wrapper_generator_bin) "$(location exaudfclient_static_bin)" "$(location exaudfclient_static)" "$(location @exaudfclient_base//:exaudfclient.template.sh)" """
FAST_BINARY_STATIC="""cp "$(location exaudfclient_static_bin)" "$(location exaudfclient_static)" """
CREATE_STATIC_BINARY_SCRIPT=select({
        "@exaudfclient_base//:fast_binary": FAST_BINARY_STATIC,
        "@exaudfclient_base//:slow_wrapper": SLOW_WRAPPER_STATIC_BINARY,
        "//conditions:default": FAST_BINARY_STATIC
    })

genrule(
    name = "exaudfclient_static",
    cmd = CREATE_STATIC_BINARY_SCRIPT,
    outs = ["exaudfclient_static"],
    srcs = [":exaudfclient_static_bin", "@exaudfclient_base//:libexaudflib_complete.so", "@exaudfclient_base//:exaudfclient.template.sh", "//:wrapper_generator_bin"],
    output_to_bindir = True
)
