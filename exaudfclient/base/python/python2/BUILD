package(default_visibility = ["//visibility:public"])

genrule(
    name = "exascript_python_tmp_cc",
    cmd = """
        INCLUDES=`$$PYTHON2_PREFIX/bin/$$PYTHON2_VERSION-config --includes`
        mkdir -p build_exascript_python_tmp_cc/exaudflib
        cp "$(location //exaudflib:script_data_transfer_objects_wrapper.h)" "$(location //exaudflib:exascript.i)" build_exascript_python_tmp_cc/exaudflib
        cd build_exascript_python_tmp_cc
        swig -v $$INCLUDES -O -DEXTERNAL_PROCESS -Wall -c++ -python -addextern -module exascript_python -o "../$(location exascript_python_tmp.cc)" exaudflib/exascript.i
        """,
    outs = ["exascript_python_tmp.cc", "exascript_python.py"],
    srcs = ["//exaudflib:exascript.i","//exaudflib:script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "exascript_python_tmp_h",
    cmd = """
        INCLUDES=`$$PYTHON2_PREFIX/bin/$$PYTHON2_VERSION-config --includes`
        mkdir build_exascript_python_tmp_h
        cp "$(location //exaudflib:script_data_transfer_objects_wrapper.h)" "$(location //exaudflib:exascript.i)" build_exascript_python_tmp_h
        cp "$(location exascript_python_tmp.cc)"  "$(location exascript_python.py)" build_exascript_python_tmp_h
        cd build_exascript_python_tmp_h
        swig -v $$INCLUDES -DEXTERNAL_PROCESS -c++ -python -external-runtime "../$(location exascript_python_tmp.h)"
        """,
    outs = ["exascript_python_tmp.h"],
    srcs = ["//exaudflib:exascript.i","//exaudflib:script_data_transfer_objects_wrapper.h", ":exascript_python_tmp.cc", "exascript_python.py"]
)

genrule(
    name = "extend_exascript_python_preset_py",
    cmd = 'bash $(location //python:extend_exascript_python_preset_py.sh) "$(location //python:exascript_python_preset_core.py)" "$(location exascript_python_preset.py)" "$$PYTHON2_PREFIX" "$$PYTHON2_VERSION" ""',
    outs = ["exascript_python_preset.py"],
    srcs = ["//python:exascript_python_preset_core.py"],
    tools = ["//python:extend_exascript_python_preset_py.sh"]
)

genrule(
    name = "exascript_python_int",
    cmd = """
      cp $(SRCS) .
      python3 $(location //:build_integrated.py) "$(location exascript_python_int.h)" "exascript_python.py" "exascript_python_wrap.py" "exascript_python_preset.py"
    """,
    outs = ["exascript_python_int.h"],
    srcs = [":exascript_python_tmp_cc", "//python:exascript_python_wrap.py", ":extend_exascript_python_preset_py"],
    tools = ["//:build_integrated.py"]
)

genrule(
    name = "filter_swig_code_exascript_python_h",
    cmd = """
        ACTUAL_PYTHON_VERSION=`$$PYTHON2_PREFIX/bin/$$PYTHON2_VERSION -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'`
        if [[ $$ACTUAL_PYTHON_VERSION == 2* ]] ; then
            python3 $(location //:filter_swig_code.py) "$@" "$<"
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
        ACTUAL_PYTHON_VERSION=`$$PYTHON2_PREFIX/bin/$$PYTHON2_VERSION -c 'import sys; print(".".join(map(str, sys.version_info[:3])))'`
        cp $(locations exascript_python_tmp_cc) .
        if [[ $$ACTUAL_PYTHON_VERSION == 2* ]] ; then
            python3 $(location //:filter_swig_code.py) "$@" exascript_python_tmp.cc
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
    deps = ["@python2//:python2","//exaudflib:exaudflib-deps","//exaudflib:header"],
    alwayslink=True,
)

cc_library(
    name = "pythoncontainer",
    srcs = ["//python:pythoncontainer.cc"],
    data = [":extend_exascript_python_preset_py"], 
    hdrs = [":exascript_python_int", ":filter_swig_code_exascript_python_h"],
    include_prefix = ".",
    deps = ["@python2//:python2",":exascript_python","//exaudflib:header",
            "//:debug_message_h","//exaudflib:scriptoptionlines", "//python:pythoncontainer_header"],
    alwayslink=True,
)
