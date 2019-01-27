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

#cp "$(location exascript_python_preset.py)" "exascript_python_preset.py"
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


# LIBS="-lpython2.7 $LIBS"
# LDFLAGS="-L$PYTHON_PREFIX/lib -Wl,-rpath,$PYTHON_PREFIX/lib $LDFLAGS" 

# echo "Compiling Python specific code"
# g++ -o exascript_python.o -c exascript_python.cc $CXXFLAGS -Wno-unused-but-set-variable || die "Failed to compile exascript_python.o"
# g++ -o pythoncontainer.o -c pythoncontainer.cc $CXXFLAGS || die "Failed to compile pythoncontainer.o"

# CONTAINER_CLIENT_OBJECT_FILES="exascript_python.o pythoncontainer.o $CONTAINER_CLIENT_OBJECT_FILES"