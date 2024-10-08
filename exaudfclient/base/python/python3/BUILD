package(default_visibility = ["//visibility:public"])

genrule(
    name = "exascript_python_tmp_cc",
    cmd = """
        INCLUDES=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION-config --includes`
        mkdir -p build_exascript_python_tmp_cc/exaudflib
        cp "$(location //base/exaudflib:swig/script_data_transfer_objects_wrapper.h)" "$(location //base/exaudflib:exascript.i)" build_exascript_python_tmp_cc/exaudflib
        cd build_exascript_python_tmp_cc
        swig -v $$INCLUDES -O -DEXTERNAL_PROCESS -Wall -c++ -python -py3 -addextern -module exascript_python -o "../$(location exascript_python_tmp.cc)" exaudflib/exascript.i
        """,
    outs = ["exascript_python_tmp.cc", "exascript_python.py"],
    srcs = ["//base/exaudflib:exascript.i","//base/exaudflib:swig/script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "exascript_python_tmp_h",
    cmd = """
        INCLUDES=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION-config --includes`
        mkdir build_exascript_python_tmp_h
        cp "$(location //base/exaudflib:swig/script_data_transfer_objects_wrapper.h)" "$(location //base/exaudflib:exascript.i)" build_exascript_python_tmp_h
        cp "$(location exascript_python_tmp.cc)"  "$(location exascript_python.py)" build_exascript_python_tmp_h
        cd build_exascript_python_tmp_h
        swig -v $$INCLUDES -DEXTERNAL_PROCESS -c++ -python -py3 -external-runtime "../$(location exascript_python_tmp.h)"
        """,
    outs = ["exascript_python_tmp.h"],
    srcs = ["//base/exaudflib:exascript.i","//base/exaudflib:swig/script_data_transfer_objects_wrapper.h", ":exascript_python_tmp.cc", "exascript_python.py"]
)

genrule(
    name = "extend_exascript_python_preset_py",
    cmd = 'bash $(location //base/python:extend_exascript_python_preset_py.sh) "$(location //base/python:exascript_python_preset_core.py)" "$(location exascript_python_preset.py)" "$$PYTHON3_PREFIX" "$$PYTHON3_VERSION" ""',
    outs = ["exascript_python_preset.py"],
    srcs = ["//base/python:exascript_python_preset_core.py"],
    tools = ["//base/python:extend_exascript_python_preset_py.sh"]
)

genrule(
    name = "exascript_python_int",
    cmd = """
      cp $(SRCS) .
      python3 $(location //base:build_integrated.py) "$(location exascript_python_int.h)" "exascript_python.py" "exascript_python_wrap.py" "exascript_python_preset.py"
    """,
    outs = ["exascript_python_int.h"],
    srcs = [":exascript_python_tmp_cc", "//base/python:exascript_python_wrap.py", ":extend_exascript_python_preset_py"],
    tools = ["//base:build_integrated.py"]
)

genrule(
    name = "filter_swig_code_exascript_python_h",
    cmd = """
        ACTUAL_PYTHON_VERSION=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'`
        if [[ $$ACTUAL_PYTHON_VERSION == 2* ]] ; then
            python3 $(location //base:filter_swig_code.py) "$@" "$<"
        else
            cp "$<" "$@" 
        fi
    """,
    outs = ["exascript_python.h"],
    srcs = [":exascript_python_tmp_h"],
    tools = ["//base:filter_swig_code.py"]
)

genrule(
    name = "filter_swig_code_exascript_python_cc",
    cmd = """
        ACTUAL_PYTHON_VERSION=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'`
        cp $(locations exascript_python_tmp_cc) .
        if [[ $$ACTUAL_PYTHON_VERSION == 2* ]] ; then
            python $(location //base:filter_swig_code.py) "$@" exascript_python_tmp.cc
        else
            cp exascript_python_tmp.cc "$@" 
        fi
    """,
    outs = ["exascript_python.cc"],
    srcs = [":exascript_python_tmp_cc"],
    tools = ["//base:filter_swig_code.py"]
)

cc_library(
    name = "exascript_python",
    srcs = [":filter_swig_code_exascript_python_cc",":filter_swig_code_exascript_python_h"],
    hdrs = [":filter_swig_code_exascript_python_h"],
    deps = ["@python3//:python3","//base/exaudflib:exaudflib-deps","//base/exaudflib:header"],
    alwayslink=True,
)

cc_library(
    name = "pythoncontainer",
    srcs = ["//base/python:pythoncontainer.cc", ":dummy.h"],
    data = [":extend_exascript_python_preset_py"], 
    hdrs = [":exascript_python_int", ":filter_swig_code_exascript_python_h"],
    include_prefix = ".",
    deps = ["@python3//:python3",":exascript_python","//base/exaudflib:header",
            "//base/utils:utils", "//base/python:pythoncontainer_header"],
    alwayslink=True,
)

#workaround to build pyextdataframe.so together with pythoncontainer c++ library
genrule(
    name = "dummy",
    cmd = 'touch "$@"',
    outs = ["dummy.h"],
    srcs = [":pyextdataframe.so"],
)

cc_binary(
    name = "pyextdataframe.so",
    linkshared = 1,
    srcs = [":python_ext_dataframe.cc"],
    deps = ["@python3//:python3","@numpy//:numpy",
           "//base/exaudflib:header", "//base/utils:utils"],
)
