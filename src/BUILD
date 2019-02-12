package(default_visibility = ["//visibility:public"])
exports_files(["filter_swig_code.py", "build_integrated.py", "exascript.i","LICENSE-exasol-script-api.txt", "debug_message.h"])

config_setting(
    name = "benchmark",
    define_values = {"benchmark": "true"},
)

config_setting(
    name = "python",
    define_values = {"python": "true"},
)

config_setting(
    name = "python3",
    define_values = {
        "python": "true",
        "python3": "true"
    },
)

config_setting(
    name = "r",
    define_values = {"r": "true"},
)

config_setting(
    name = "java",
    define_values = {"java": "true"},
)


BENCHMARK_VM_ENABLED_DEFINE=select({
        ":benchmark": ["ENABLE_BENCHMARK_VM"],
        "//conditions:default": []
    }) 
PYTHON_VM_ENABLED_DEFINE=select({
        ":python": ["ENABLE_PYTHON_VM"],
        "//conditions:default": []
    })
R_VM_ENABLED_DEFINE=select({
        ":r": ["ENABLE_R_VM"],
        "//conditions:default": []
    })
JAVA_VM_ENABLED_DEFINE=select({
        ":java": ["ENABLE_JAVA_VM"],
        "//conditions:default": []
    })

VM_ENABLED_DEFINES=BENCHMARK_VM_ENABLED_DEFINE+PYTHON_VM_ENABLED_DEFINE+R_VM_ENABLED_DEFINE+JAVA_VM_ENABLED_DEFINE

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
    srcs = ["exaudflib.cc","exaudflib.h","debug_message.h"],
    hdrs = ["exaudflib.h"],
    linkstatic = 1,
    defines = VM_ENABLED_DEFINES,
    deps = [":script_data_transfer_objects_wrapper", ":zmqcontainer", "@zmq//:zmqlib"],
)

cc_binary(
    name = "libexaudflib_complete.so",
    linkshared = 1,
    srcs = [":exaudflib", ":zmqcontainer", ":script_data_transfer_objects_wrapper", ":script_data_transfer_objects"],
    deps = [ "@zmq//:zmqlib"],
    defines = VM_ENABLED_DEFINES
)

VM_ENABLED_DEPS=select({
        ":benchmark": [":benchmark_container"],
        "//conditions:default": []
    }) + select({
        ":python": ["//python:pythoncontainer"],
        "//conditions:default": []
    }) + select({
        ":r": ["//rcontainer:rcontainer"],
        "//conditions:default": []
    }) + select({
        ":java": ["//javacontainer:javacontainer"],
        "//conditions:default": []
    })

cc_binary(
    name = "exaudfclient",
    srcs = ["exaudfclient.cc","exaudflib.h",
            "script_data_transfer_objects_wrapper.h", 
            "script_data_transfer_objects.h", 
            "debug_message.h"],
    linkopts = ["-ldl"],
    deps = VM_ENABLED_DEPS,
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"] + select({
            "python3": ["//python:pyextdataframe.so"],
            "//conditions:default": []
        })
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
