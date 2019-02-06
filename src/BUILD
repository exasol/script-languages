package(default_visibility = ["//visibility:public"])
exports_files(["filter_swig_code.py", "build_integrated.py", "exascript.i"])

# load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

# proto_library(
#     name = "zmqcontainer_proto",
#     srcs = ["zmqcontainer.proto"],
#     deps = ["@com_google_protobuf//:any_proto"],
# )

#load("@build_stack_rules_proto//cpp:cpp_proto_compile.bzl", "cpp_proto_compile")
# load("@build_stack_rules_proto//cpp:cpp_proto_library.bzl", "cpp_proto_library")
# cpp_proto_library(
#     name = "zmqcontainer",
#     deps = [":zmqcontainer_proto"]
# )

genrule(
    name = "zmqcontainer_proto",
    cmd = """$$CUSTOM_PROTOBUF_BIN -I. $< --cpp_out=. &&
            cp zmqcontainer.pb.cc $(location zmqcontainer.pb.cc) &&
            cp zmqcontainer.pb.h $(location zmqcontainer.pb.h)""",
    outs = ["zmqcontainer.pb.cc","zmqcontainer.pb.h"],
    srcs = ["zmqcontainer.proto"]
)

cc_library(
    name = "zmqcontainer",
    hdrs = [":zmqcontainer_proto"],
    srcs = [":zmqcontainer_proto"],
)

cc_library(
    name = "scriptoptionlines",
    hdrs = ["scriptoptionlines.h"],
    srcs = ["scriptoptionlines.cc","scriptoptionlines.h"],
)


cc_library(
    name = "script_data_transfer_objects",
    srcs = ["script_data_transfer_objects.cc","script_data_transfer_objects.h"],
    hdrs = ["script_data_transfer_objects.h"],
)

cc_library(
    name = "script_data_transfer_objects_wrapper",
    srcs = ["script_data_transfer_objects_wrapper.cc","script_data_transfer_objects_wrapper.h"],
    deps = [":script_data_transfer_objects"],
    hdrs = ["script_data_transfer_objects_wrapper.h"]
)

cc_library(
    name = "exaudflib",
    srcs = ["exaudflib.cc","exaudflib.h"],
    hdrs = ["exaudflib.h"],
    linkstatic = 1,
    deps = [":script_data_transfer_objects_wrapper", ":zmqcontainer", "@zmq//:zmqlib"],
)

cc_binary(
    name = "libexaudflib_complete.so",
    linkshared = 1,
    srcs = [":exaudflib", ":zmqcontainer", ":script_data_transfer_objects_wrapper", ":script_data_transfer_objects"],
    deps = [ "@zmq//:zmqlib"],
)

cc_binary(
    name = "exaudfclient",
    srcs = ["exaudfclient.cc","exaudflib.h","script_data_transfer_objects_wrapper.h", "script_data_transfer_objects.h", "debug_message.h"],
    linkopts = ["-ldl"],
    deps = [":benchmark_container", ":streamingcontainer", "//python/python2:python2_7_pythoncontainer", "//rcontainer:rcontainer"],
    #defines = ["CUSTOM_PROTOBUF_PREFIX=\"%s\""%CUSTOM_PROTOBUF_PREFIX]
    data = [":libexaudflib_complete.so"]
)

cc_library(
    name = "streamingcontainer",
    srcs = ["streamingcontainer.cc","streamingcontainer.h"],
    deps = [":exaudflib"]
)

cc_library(
    name = "benchmark_container",
    srcs = ["benchmark_container.cc","benchmark_container.h", "debug_message.h"],
    hdrs = ["benchmark_container.h"],
    deps = [":exaudflib"]
)