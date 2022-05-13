package(default_visibility = ["//visibility:public"])
exports_files(["exascript.i"])

load("//:variables.bzl", "VM_ENABLED_DEFINES")

cc_library(
    name = "scriptoptionlines",
    hdrs = ["vm/scriptoptionlines.h"],
    srcs = ["vm/scriptoptionlines.cc","vm/scriptoptionlines.h"],
    copts= ["-fno-lto"],
    # We deactivate link time optimization, because otherwise we get the following linker/compiler error while linking/compiling exaudfclient
    # function ExecutionGraph::extractOptionLine(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, unsigned long&, std::function<void (char const*)>): error: undefined reference to 'std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > std::operator+<char, std::char_traits<char>, std::allocator<char> >(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, char const*)'
    alwayslink=True,
)

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
    alwayslink=True,
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
    alwayslink=True,
)

# A collection target for the case that language container want to include all public headers of exaudflib
cc_library(
    name = "header",
    hdrs = [
        "udf_plugin_interface.h", "load_dynamic.h", "vm/swig_vm.h",
        "swig/swig_common.h", "swig/swig_meta_data.h", "swig/swig_result_handler.h", "swig/swig_table_iterator.h"
    ],
    alwayslink=True,
)

# A collection target for the case that language container want to depent on all public libraries of exaudflib
# which won't leak protobuf through linker namespaces. Caution: Can't be used for the target libexaudflib-complete.so,
# because this target declares the libraries only as deps.
cc_library( #TODO might be not working as expected
    name = "exaudflib-deps",
    deps = [":script_data_transfer_objects",":script_data_transfer_objects_wrapper"],
    alwayslink=True,
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
    alwayslink=True,
)

# Do not depend on this library in the langauge container or exaudfclient. 
# The only target which must depend on it is //:libexaudflib-complete.so.
# If you depend on this target you will load protobuf into your linker namespace
# which might cause problems with tensorflow. If you need
# "//exaudflib:scriptoptionlines","//exaudflib:script_data_transfer_objects" or
# "//exaudflib:script_data_transfer_objects_wrapper" in the language container 
# depend directly on them
cc_library(
    name = "exaudflib",
    srcs = ["vm/swig_vm.cc",
            "swig/swig_common.cc", "swig/swig_meta_data.cc", "swig/swig_result_handler.cc",
            "swig/swig_table_iterator.cc",
            "impl/exaudflib_main.cc",
            "impl/msg_conversion.cc", "impl/msg_conversion.h",
            "impl/check.cc", "impl/check.h",
            "impl/socket_info.cc", "impl/socket_info.h",
            "impl/socket_low_level.cc", "impl/socket_low_level.h",
            "impl/global.cc", "impl/global.h",
            "impl/socket_high_level.cc", "impl/socket_high_level.h",
            "impl/swig/swig_factory.cc", "impl/swig/swig_parameter.cc", "impl/swig/swig_parameter.h",
            "impl/swig/swig_table_iterator.h", "impl/swig/swig_table_iterator.cc",
            "impl/swig/swig_meta_data.h", "impl/swig/swig_meta_data.cc",
            "impl/swig/swig_result_handler.h", "impl/swig/swig_result_handler.cc",
            "impl/swig/swig_general_iterator.h", "impl/swig/swig_general_iterator.cc"],
    linkstatic = 1,
    alwayslink = True,
    defines = VM_ENABLED_DEFINES,
    deps = [":exaudflib-deps", ":zmqcontainer", "@zmq//:zmqlib", ":header", "//:debug_message_h"],
)
