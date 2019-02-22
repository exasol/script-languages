package(default_visibility = ["//visibility:public"])

genrule(
    name = "exascript_python_tmp_cc",
    cmd = """
        INCLUDES=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION-config --includes`
        swig -v $$INCLUDES -O -DEXTERNAL_PROCESS -Wall -c++ -python -py3 -addextern -module exascript_python -o "$(location exascript_python_tmp.cc)" "$(location //exaudflib:exascript.i)"
        """,
    outs = ["exascript_python_tmp.cc", "exascript_python.py"],
    srcs = ["//exaudflib:exascript.i","//exaudflib:script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "exascript_python_tmp_h",
    cmd = """
        INCLUDES=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION-config --includes`
        swig -v $$INCLUDES -DEXTERNAL_PROCESS -c++ -python -py3 -external-runtime "$(location exascript_python_tmp.h)"
        """,
    outs = ["exascript_python_tmp.h"],
    srcs = ["//exaudflib:exascript.i","//exaudflib:script_data_transfer_objects_wrapper.h", ":exascript_python_tmp_cc"]
)

genrule(
    name = "extend_exascript_python_preset_py",
    cmd = 'bash $(location //python:extend_exascript_python_preset_py.sh) "$(location //python:exascript_python_preset_core.py)" "$(location exascript_python_preset.py)" "$$PYTHON3_PREFIX" "$$PYTHON3_VERSION" ""',
    outs = ["exascript_python_preset.py"],
    srcs = ["//python:exascript_python_preset_core.py"],
    tools = ["//python:extend_exascript_python_preset_py.sh"]
)

genrule(
    name = "exascript_python_int",
    cmd = 'cp $(SRCS) . && python build_integrated.py "$(location exascript_python_int.h)" "exascript_python.py" "exascript_python_wrap.py" "exascript_python_preset.py"',
    outs = ["exascript_python_int.h"],
    srcs = [":exascript_python_tmp_cc", "//python:exascript_python_wrap.py", ":extend_exascript_python_preset_py"],
    tools = ["//:build_integrated.py"]
)

genrule(
    name = "filter_swig_code_exascript_python_h",
    cmd = """
        ACTUAL_PYTHON_VERSION=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'`
        if [[ $$ACTUAL_PYTHON_VERSION == 2* ]] ; then
            python ./filter_swig_code.py "$@" "$<"
        else
            cp "$<" "$@" 
        fi
    """,
    outs = ["exascript_python.h"],
    srcs = [":exascript_python_tmp_h"],
    tools = ["//:filter_swig_code.py"]
)

genrule(
    name = "filter_swig_code_exascript_python_cc",
    cmd = """
        ACTUAL_PYTHON_VERSION=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'`
        cp $(locations exascript_python_tmp_cc) .
        if [[ $$ACTUAL_PYTHON_VERSION == 2* ]] ; then
            python ./filter_swig_code.py "$@" exascript_python_tmp.cc
        else
            cp exascript_python_tmp.cc "$@" 
        fi
    """,
    outs = ["exascript_python.cc"],
    srcs = [":exascript_python_tmp_cc"],
    tools = ["//:filter_swig_code.py"]
)

cc_library(
    name = "exascript_python",
    srcs = [":filter_swig_code_exascript_python_cc",":filter_swig_code_exascript_python_h"],
    hdrs = [":filter_swig_code_exascript_python_h"],
    deps = ["@python3//:python3","//exaudflib:exaudflib"]
)

cc_library(
    name = "pythoncontainer",
    srcs = ["//python:pythoncontainer.cc","//:debug_message.h",":dummy.h"],
    data = [":extend_exascript_python_preset_py"], 
    hdrs = [":exascript_python_int", ":filter_swig_code_exascript_python_h"],
    include_prefix = ".",
    deps = ["@python3//:python3","//exaudflib:exaudflib","//exaudflib:scriptoptionlines",":exascript_python"]
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
    srcs = ["//python:python_ext_dataframe.cc","//:debug_message.h",
            "//exaudflib:exaudflib.h","//exaudflib:script_data_transfer_objects_wrapper.h",
            "//exaudflib:script_data_transfer_objects.h"],
    deps = ["@python3//:python3","@numpy_linux//:numpy_linux"]
)
