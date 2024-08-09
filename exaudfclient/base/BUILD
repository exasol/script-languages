
package(default_visibility = ["//visibility:public"])
exports_files(["filter_swig_code.py", "build_integrated.py",
	       "create_binary_wrapper.sh", "create_binary_wrapper_valgrind.sh",
	       "create_binary_wrapper_valgrind_massif.sh", "exaudfclient.template.sh"])

load("//:variables.bzl", "VM_ENABLED_DEFINES")

cc_library(
    name = "debug_message_h",
    hdrs = [
        "debug_message.h"
    ],
)

cc_library(
    name = "load_dynamic",
    srcs = [
        "load_dynamic.cc"
    ],
    deps = ["//base/exaudflib:header", "//base:debug_message_h", "//base/exaudflib:exaudflib-deps"],
    defines = VM_ENABLED_DEFINES,
)


# Using cc_binary to build a shared library as output target is a workaround, because
# Bazel's cc_library are only intermediate stages and can be used as output target.
# It is necessary to include //base/exaudflib:exaudflib into deps and srcs, because
# otherwise dlmopen won't find the symbols of exaudflib and its dependencies. The target //base/exaudflib:exaudflib-deps
# won't work either, because it only contains the libraries as dependencies.
# Bazel builts a static library (.a) and a dynamic library (.so) out of exaudflib:exaudflib.
# In the libexaudflib_complete.so-2.params, we saw that Bazel uses both libraries to built libexaudflib_complete.so.
# Experiments have shown that we get missing symbols during loading libexaudflib_complete.so, if we only built the static library.
cc_binary(
    name = "libexaudflib_complete.so",
    linkshared = 1,
    srcs = ["//base/exaudflib:exaudflib"],
    deps = ["//base/exaudflib:exaudflib"],
    defines = VM_ENABLED_DEFINES,
)
