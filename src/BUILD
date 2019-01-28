load("@org_pubref_rules_protobuf//cpp:rules.bzl", "cc_proto_library")


cc_proto_library(
    name = "zmqcontainer",
    protos = ["zmqcontainer.proto"],
    with_grpc = False
)

cc_library(
    name = "scriptoptionlines",
    hdrs = ["scriptoptionlines.h"],
    srcs = ["scriptoptionlines.cc","scriptoptionlines.h"]
)


cc_library(
    name = "script_data_transfer_objects",
    srcs = ["script_data_transfer_objects.cc","script_data_transfer_objects.h"],
    hdrs = ["script_data_transfer_objects.h"]
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
    deps = [":script_data_transfer_objects_wrapper", ":zmqcontainer"]
)

cc_library(
    name = "exaudfclient",
    srcs = ["exaudfclient.cc"],
    deps = [":exaudflib"]
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

genrule(
    name = "python2_7_exascript_python_tmp_cc",
    cmd = 'swig -v -I/usr/include/python2.7 -O -DEXTERNAL_PROCESS -Wall -c++ -python -addextern -module exascript_python -o "$(location exascript_python_tmp.cc)" "$(location exascript.i)"',
    outs = ["exascript_python_tmp.cc"],
    srcs = ["exascript.i","script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "python2_7_exascript_python_tmp_h",
    cmd = 'swig -v -I/usr/include/python2.7 -DEXTERNAL_PROCESS -c++ -python -external-runtime "$(location exascript_python_tmp.h)"',
    outs = ["exascript_python_tmp.h"],
    srcs = ["exascript.i","script_data_transfer_objects_wrapper.h", ":python2_7_exascript_python_tmp_cc"]
)


genrule(
    name = "python2_7_extend_exascript_python_preset_py",
    cmd = 'bash $(location extend_exascript_python_preset_py.sh) "$(location exascript_python_preset_core.py)" "$(location exascript_python_preset.py)" "/usr"',
    outs = ["exascript_python_preset.py"],
    srcs = ["exascript_python_preset_core.py"],
    tools = ["extend_exascript_python_preset_py.sh"]
)

genrule(
    name = "python2_7_exascript_python_int",
    cmd = 'cp "$(location python2_7_extend_exascript_python_preset_py)" "exascript_python_preset.py" && python ./build_integrated.py "$(location exascript_python_int.h)" "exascript_python.py" "exascript_python_wrap.py" "exascript_python_preset.py"',
    outs = ["exascript_python_int.h"],
    srcs = ["exascript_python.py", "exascript_python_wrap.py", ":python2_7_extend_exascript_python_preset_py"],
    tools = ["build_integrated.py"]
)

genrule(
    name = "python2_7_filter_swig_code_exascript_python_h",
    cmd = 'python ./filter_swig_code.py "$@" "$<"',
    outs = ["exascript_python.h"],
    srcs = [":python2_7_exascript_python_tmp_h"],
    tools = ["filter_swig_code.py"]
)

genrule(
    name = "python2_7_filter_swig_code_exascript_python_cc",
    cmd = 'python ./filter_swig_code.py "$@" "$<"',
    outs = ["exascript_python.cc"],
    srcs = [":python2_7_exascript_python_tmp_cc"],
    tools = ["filter_swig_code.py"]
)


cc_library(
    name = "python2_7_exascript_python",
    srcs = [":python2_7_filter_swig_code_exascript_python_cc",":python2_7_filter_swig_code_exascript_python_h"],
    hdrs = [":python2_7_filter_swig_code_exascript_python_h"],
    copts = ["-DENABLE_PYTHON_VM"],
    deps = ["@python_2_7_linux//:python27-lib",":exaudflib"]
)

cc_library(
    name = "python2_7_pythoncontainer",
    srcs = ["pythoncontainer.cc",":python2_7_exascript_python_int"],
    data = [":python2_7_extend_exascript_python_preset_py"],
    copts = ["-DENABLE_PYTHON_VM"],
    deps = ["@python_2_7_linux//:python27-lib",":exaudflib",":python2_7_exascript_python", ":scriptoptionlines"]
)

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