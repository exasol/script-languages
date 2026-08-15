package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

genrule(
    name = "exascript_python_tmp_cc",
    cmd = """
        INCLUDES=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION-config --includes`
        mkdir -p build_exascript_python_tmp_cc/exaudflib
        cp "$(location //exaudflib:swig/script_data_transfer_objects_wrapper.h)" "$(location //exaudflib:exascript.i)" build_exascript_python_tmp_cc/exaudflib
        cd build_exascript_python_tmp_cc
        swig -v $$INCLUDES -O -DEXTERNAL_PROCESS -Wall -c++ -python -py3 -addextern -module exascript_python -o "../$(location exascript_python.cc)" exaudflib/exascript.i
        #Run replace_swig_import_helper.py because the generated exascript_python.py is not compatible with Python 3.12
        #TODO: Update Swig to a new version which produces code compatible with Python 3.12
        cd ..
        $$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION "$(location //:replace_swig_import_helper.py)" "$(location exascript_python.py)" "$(location exascript_python.py)"
        """,
    outs = ["exascript_python.cc", "exascript_python.py"],
    srcs = ["//exaudflib:exascript.i","//exaudflib:swig/script_data_transfer_objects_wrapper.h"],
    tools = ["//:replace_swig_import_helper.py"],
)

genrule(
    name = "exascript_python_tmp_h",
    cmd = """
        INCLUDES=`$$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION-config --includes`
        swig -v $$INCLUDES -DEXTERNAL_PROCESS -c++ -python -py3 -external-runtime "$(location exascript_python.h)"
        """,
    outs = ["exascript_python.h"],
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
    cmd = """
      cp $(SRCS) .
      $$PYTHON3_PREFIX/bin/$$PYTHON3_VERSION $(location //:build_integrated.py) "$(location exascript_python_int.h)" "exascript_python.py" "exascript_python_wrap.py" "exascript_python_preset.py"
    """,
    outs = ["exascript_python_int.h"],
    srcs = [":exascript_python_tmp_cc", "//python:exascript_python_wrap.py", ":extend_exascript_python_preset_py"],
    tools = ["//:build_integrated.py"]
)

cc_library(
    name = "exascript_python",
    srcs = [":exascript_python_tmp_cc", ":exascript_python_tmp_h"],
    hdrs = [":exascript_python_tmp_h"],
    deps = ["@python3//:python3","//exaudflib:exaudflib-deps","//exaudflib:header"],
    alwayslink=True,
)

cc_library(
    name = "pythoncontainer",
    srcs = ["//python:pythoncontainer.cc", ":dummy.h"],
    data = [":extend_exascript_python_preset_py"], 
    hdrs = [":exascript_python_int", ":exascript_python_tmp_h"],
    include_prefix = ".",
    deps = ["@python3//:python3",":exascript_python","//exaudflib:header",
            "//utils:utils", "//python:pythoncontainer_header"],
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
           "//exaudflib:header", "//utils:utils"],
)
