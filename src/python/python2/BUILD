package(default_visibility = ["//visibility:public"])

genrule(
    name = "python2_7_exascript_python_tmp_cc",
    cmd = 'swig -v -I$$PYTHON_PREFIX/include/python2.7 -O -DEXTERNAL_PROCESS -Wall -c++ -python -addextern -module exascript_python -o "$(location exascript_python_tmp.cc)" "$(location //:exascript.i)"',
    outs = ["exascript_python_tmp.cc"],
    srcs = ["//:exascript.i","//:script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "python2_7_exascript_python_tmp_h",
    cmd = 'swig -v -I$$PYTHON_PREFIX/include/python2.7 -DEXTERNAL_PROCESS -c++ -python -external-runtime "$(location exascript_python_tmp.h)"',
    outs = ["exascript_python_tmp.h"],
    srcs = ["//:exascript.i","//:script_data_transfer_objects_wrapper.h", ":python2_7_exascript_python_tmp_cc"]
)


genrule(
    name = "python2_7_extend_exascript_python_preset_py",
    cmd = 'bash $(location //python:extend_exascript_python_preset_py.sh) "$(location //python:exascript_python_preset_core.py)" "$(location exascript_python_preset.py)" "$$PYTHON_PREFIX"',
    outs = ["exascript_python_preset.py"],
    srcs = ["//python:exascript_python_preset_core.py"],
    tools = ["//python:extend_exascript_python_preset_py.sh"]
)

genrule(
    name = "python2_7_exascript_python_int",
    cmd = 'cp $(SRCS) . && python build_integrated.py "$(location exascript_python_int.h)" "exascript_python.py" "exascript_python_wrap.py" "exascript_python_preset.py"',
    outs = ["exascript_python_int.h"],
    srcs = ["//python:exascript_python.py", "//python:exascript_python_wrap.py", ":python2_7_extend_exascript_python_preset_py"],
    tools = ["//:build_integrated.py"]
)

genrule(
    name = "python2_7_filter_swig_code_exascript_python_h",
    cmd = 'python ./filter_swig_code.py "$@" "$<"',
    outs = ["exascript_python.h"],
    srcs = [":python2_7_exascript_python_tmp_h"],
    tools = ["//:filter_swig_code.py"]
)

genrule(
    name = "python2_7_filter_swig_code_exascript_python_cc",
    cmd = 'python ./filter_swig_code.py "$@" "$<"',
    outs = ["exascript_python.cc"],
    srcs = [":python2_7_exascript_python_tmp_cc"],
    tools = ["//:filter_swig_code.py"]
)

cc_library(
    name = "python2_7_exascript_python",
    srcs = [":python2_7_filter_swig_code_exascript_python_cc"],
    copts = ["-DENABLE_PYTHON_VM"],
    deps = ["@python_2_7_linux//:test","//:exaudflib"]
)

cc_library(
    name = "python2_7_pythoncontainer",
    srcs = ["//python:pythoncontainer.cc"],
    data = [":python2_7_extend_exascript_python_preset_py"], 
    copts = ["-DENABLE_PYTHON_VM"],
    hdrs = [":python2_7_exascript_python_int", ":python2_7_filter_swig_code_exascript_python_h"],
    include_prefix = ".",
    deps = ["@python_2_7_linux//:test","//:exaudflib","//:scriptoptionlines",":python2_7_exascript_python"]
)