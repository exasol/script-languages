
package(default_visibility = ["//visibility:public"])
exports_files(["filter_swig_code.py", "build_integrated.py",
	       "create_binary_wrapper.sh", "create_binary_wrapper_valgrind.sh",
	       "create_binary_wrapper_valgrind_massif.sh", "exaudfclient.template.sh",
	       "replace_swig_import_helper.py"])

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
    name = "load_dynamic",
    srcs = [
        "load_dynamic.cc"
    ],
    deps = ["//exaudflib:header", "//utils:utils", "//exaudflib:exaudflib-deps"],
    defines = VM_ENABLED_DEFINES,
)


# Using cc_binary to build a shared library as output target is a workaround, because
# Bazel's cc_library are only intermediate stages and can't be used as output target.
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
