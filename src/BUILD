package(default_visibility = ["//visibility:public"])
# load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")

proto_library(
    name = "zmqcontainer_proto",
    srcs = ["zmqcontainer.proto"],
    deps = ["@com_google_protobuf//:any_proto"],
)

load("@build_stack_rules_proto//cpp:cpp_proto_compile.bzl", "cpp_proto_compile")
load("@build_stack_rules_proto//cpp:cpp_proto_library.bzl", "cpp_proto_library")
cpp_proto_library(
    name = "zmqcontainer",
    deps = [":zmqcontainer_proto"]
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

CUSTOM_PROTOBUF_PREFIX='$$CUSTOM_PROTOBUF_PREFIX'

cc_library(
    name = "exaudflib",
    srcs = ["exaudflib.cc","exaudflib.h"],
    hdrs = ["exaudflib.h"],
    deps = [":script_data_transfer_objects_wrapper", ":zmqcontainer"],
    defines = ["CUSTOM_PROTOBUF_PREFIX=\"%s\""%CUSTOM_PROTOBUF_PREFIX]
)

cc_library(
    name = "exaudfclient",
    srcs = ["exaudfclient.cc"],
    deps = [":exaudflib"],
    defines = ["CUSTOM_PROTOBUF_PREFIX=\"%s\""%CUSTOM_PROTOBUF_PREFIX]
)

cc_library(
    name = "streamingcontainer",
    srcs = ["streamingcontainer.cc","streamingcontainer.h"],
    deps = [":exaudflib"]

)

cc_library(
    name = "benchmark_container",
    srcs = ["benchmark_container.cc","benchmark_container.h"],
    hdrs = ["benchmark_container.h"],
    deps = [":exaudflib"]
)


##########################################################


##########################################################

genrule(
    name = "exascript_r_tmp_cc",
    cmd = 'swig -v -O -DEXTERNAL_PROCESS -Wall -c++ -r -addextern -module exascript_r -o "$(location exascript_r_tmp.cc)" "$(location exascript.i)"',
    outs = ["exascript_r_tmp.cc","exascript_r.R"],
    srcs = ["exascript.i","script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "exascript_r_tmp_h",
    cmd = 'swig -DEXTERNAL_PROCESS -c++ -r -external-runtime "$(location exascript_r_tmp.h)"',
    outs = ["exascript_r_tmp.h"],
    srcs = ["exascript.i","script_data_transfer_objects_wrapper.h", ":python2_7_exascript_python_tmp_cc"]
)

genrule(
    name = "exascript_r_int",
    cmd = 'cp $(locations :exascript_r_tmp_cc) . && python ./build_integrated.py "$(location exascript_r_int.h)" exascript_r.R"" "exascript_r_wrap.R" "exascript_r_preset.R"',
    outs = ["exascript_r_int.h"],
    srcs = ["exascript_r_wrap.R", "exascript_r_preset.R", ":exascript_r_tmp_cc"],
    tools = ["build_integrated.py"]
)

genrule(
    name = "filter_swig_code_exascript_r_h",
    cmd = 'python ./filter_swig_code.py "$@" "$<"',
    outs = ["exascript_r.h"],
    srcs = [":exascript_r_tmp_h"],
    tools = ["filter_swig_code.py"]
)

genrule(
    name = "filter_swig_code_exascript_r_cc",
    cmd = 'cp $(locations :exascript_r_tmp_cc) . && python ./filter_swig_code.py "$@" "exascript_r_tmp.cc"',
    outs = ["exascript_r.cc"],
    srcs = [":exascript_r_tmp_cc"],
    tools = ["filter_swig_code.py"]
)


cc_library(
    name = "exascript_r",
    srcs = [":filter_swig_code_exascript_r_cc",":filter_swig_code_exascript_r_h"],
    hdrs = [":filter_swig_code_exascript_r_h"],
    copts = ["-DENABLE_R_VM"],
    deps = ["@R_linux//:r-lib",":exaudflib"]
)

cc_library(
    name = "rcontainer",
    srcs = ["rcontainer.cc",":exascript_r_int"],
    data = [":python2_7_extend_exascript_python_preset_py"],
    copts = ["-DENABLE_R_VM"],
    deps = ["@R_linux//:r-lib",":exaudflib",":exascript_r", ":scriptoptionlines"]
)



genrule(
  name = "g",
  srcs = [],
  outs = ["g.out"],
  cmd = "echo FOO is equal to $$PYTHON_PREFIX! > $@",
  #toolchains = [":set_foo_to_bar"]
)