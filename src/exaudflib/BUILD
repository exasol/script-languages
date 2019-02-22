package(default_visibility = ["//visibility:public"])
exports_files(["exaudflib.h","exascript.i"])

load("//:variables.bzl", "VM_ENABLED_DEFINES")

genrule(
    name = "zmqcontainer_proto",
    cmd = """
            TMPDIR=`mktemp -d`
            cp $< "$$TMPDIR"
            (cd "$$TMPDIR"
            $$CUSTOM_PROTOBUF_BIN -I. zmqcontainer.proto --cpp_out=.)
            cp "$$TMPDIR"/zmqcontainer.pb.cc $(location zmqcontainer.pb.cc)
            cp "$$TMPDIR"/zmqcontainer.pb.h $(location zmqcontainer.pb.h)""",
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
    srcs = ["exaudflib.cc","exaudflib.h","//:debug_message.h"],
    hdrs = ["exaudflib.h"],
    linkstatic = 1,
    defines = VM_ENABLED_DEFINES,
    deps = [":script_data_transfer_objects_wrapper", ":zmqcontainer", "@zmq//:zmqlib"],
)
