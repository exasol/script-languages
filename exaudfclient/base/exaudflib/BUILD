package(default_visibility = ["//visibility:public"])
exports_files(["exascript.i"])

load("//:variables.bzl", "VM_ENABLED_DEFINES")

cc_library(
    name = "script_data_transfer_objects",
    srcs = ["swig/script_data_transfer_objects.cc","swig/script_data_transfer_objects.h"],
    hdrs = ["swig/script_data_transfer_objects.h"],
    copts= select({
        "//:benchmark": ["-fno-lto"],
        "//conditions:default": []
    }),
    # We deactivate link time optimization for benchmark,
    # because otherwise we get linker/compiler error while linking/compiling exaudfclient
    # Benchmark causes this, because it doesn't refer it its implementation.
)

cc_library(
    name = "script_data_transfer_objects_wrapper",
    srcs = ["swig/script_data_transfer_objects_wrapper.cc","swig/script_data_transfer_objects_wrapper.h"],
    deps = [":script_data_transfer_objects"],
    hdrs = ["swig/script_data_transfer_objects_wrapper.h"],
    copts= select({
        "//:benchmark": ["-fno-lto"],
        "//conditions:default": []
    }),
    # We deactivate link time optimization for benchmark,
    # because otherwise we get linker/compiler error while linking/compiling exaudfclient
    # Benchmark causes this, because it doesn't refer it its implementation.
)

# A collection target for the case that language container want to include all public headers of exaudflib
cc_library(
    name = "header",
    hdrs = [
        "udf_plugin_interface.h", "load_dynamic.h", "vm/swig_vm.h",
        "swig/swig_common.h", "swig/swig_meta_data.h", "swig/swig_result_handler.h", "swig/swig_table_iterator.h"
    ],
)

# A collection target for the case that language container want to depent on all public libraries of exaudflib
# which won't leak protobuf through linker namespaces. Caution: Can't be used for the target libexaudflib-complete.so,
# because this target declares the libraries only as deps.
cc_library( #TODO might be not working as expected
    name = "exaudflib-deps",
    deps = [":script_data_transfer_objects",":script_data_transfer_objects_wrapper"],
)

genrule(
    name = "zmqcontainer_proto",
    cmd = """
            TMPDIR=`mktemp -d`
            cp $< "$$TMPDIR"
            (cd "$$TMPDIR"
            $$PROTOBUF_BIN -I. zmqcontainer.proto --cpp_out=.)
            cp "$$TMPDIR"/zmqcontainer.pb.cc $(location zmqcontainer.pb.cc)
            cp "$$TMPDIR"/zmqcontainer.pb.h $(location zmqcontainer.pb.h)""",
    outs = ["zmqcontainer.pb.cc","zmqcontainer.pb.h"],
    srcs = ["zmqcontainer.proto"]
)

cc_library(
    name = "zmqcontainer",
    hdrs = [":zmqcontainer_proto"],
    srcs = [":zmqcontainer_proto"],
    deps = ["@protobuf//:protobuf"],
    copts = ["-DNDEBUG=1"] #We use the protobuf shared library compiled  with NDEBUG=1 (the conda package is the release version). We need to set NDEBUG=1, otherwise some functions are not visisble for the dependent libraries (compiled inline for NDEBUG=1, compiled as global function otherwise).

)

# Do not depend on this library in the langauge container or exaudfclient.
# The only target which must depend on it is //:libexaudflib-complete.so.
# If you depend on this target you will load protobuf into your linker namespace
# which might cause problems with tensorflow. If you need
# "//base/exaudflib:script_data_transfer_objects" or
# "//base/exaudflib:script_data_transfer_objects_wrapper" in the language container
# depend directly on them or depend on exaudflib-deps
cc_library(
    name = "exaudflib",
    srcs = ["vm/swig_vm.cc",
            "swig/swig_common.cc", "swig/swig_meta_data.cc", "swig/swig_result_handler.cc", "swig/swig_table_iterator.cc",
            "impl/exaudflib_main.cc",
            "impl/msg_conversion.cc", "impl/msg_conversion.h",
            "impl/check.cc", "impl/check.h",
            "impl/socket_info.cc", "impl/socket_info.h",
            "impl/socket_low_level.cc", "impl/socket_low_level.h",
            "impl/global.cc", "impl/global.h",
            "impl/socket_high_level.cc", "impl/socket_high_level.h",
            "impl/swig/swig_factory.cc",
            "impl/swig/swig_table_iterator.h", "impl/swig/swig_table_iterator.cc",
            "impl/swig/swig_meta_data.h", "impl/swig/swig_meta_data.cc",
            "impl/swig/swig_result_handler.h", "impl/swig/swig_result_handler.cc",
            "impl/swig/swig_general_iterator.h", "impl/swig/swig_general_iterator.cc"],
    defines = VM_ENABLED_DEFINES,
    linkstatic = False, # Needs to be false, because otherwise we might get missing symbols when loading exaudflib_complete.so
    deps = [":exaudflib-deps", ":zmqcontainer", "@zmq//:zmq", ":header", "//base/utils:utils"],
)
