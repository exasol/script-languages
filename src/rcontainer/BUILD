package(default_visibility = ["//visibility:public"])

genrule(
    name = "exascript_r_tmp_cc",
    cmd = 'swig -v -O -DEXTERNAL_PROCESS -Wall -c++ -r -addextern -module exascript_r -o "$(location exascript_r_tmp.cc)" "$(location //exaudflib:exascript.i)"',
    outs = ["exascript_r_tmp.cc","exascript_r.R"],
    srcs = ["//exaudflib:exascript.i","//exaudflib:script_data_transfer_objects_wrapper.h"]
)

genrule(
    name = "exascript_r_tmp_h",
    cmd = 'swig -DEXTERNAL_PROCESS -c++ -r -external-runtime "$(location exascript_r_tmp.h)"',
    outs = ["exascript_r_tmp.h"],
    srcs = ["//exaudflib:exascript.i","//exaudflib:script_data_transfer_objects_wrapper.h", ":exascript_r_tmp_cc"]
)

genrule(
    name = "exascript_r_int",
    cmd = """
            cp $(locations :exascript_r_tmp_cc) $(location exascript_r_wrap.R) $(location exascript_r_preset.R) . && 
            python ./build_integrated.py "$(location exascript_r_int.h)" exascript_r.R"" "exascript_r_wrap.R" "exascript_r_preset.R"
        """,
    outs = ["exascript_r_int.h"],
    srcs = ["exascript_r_wrap.R", "exascript_r_preset.R", ":exascript_r_tmp_cc"],
    tools = ["//:build_integrated.py"]
)

genrule(
    name = "filter_swig_code_exascript_r_h",
    cmd = 'python ./filter_swig_code.py "$@" "$<"',
    outs = ["exascript_r.h"],
    srcs = [":exascript_r_tmp_h"],
    tools = ["//:filter_swig_code.py"]
)

genrule(
    name = "filter_swig_code_exascript_r_cc",
    cmd = """
            cp $(locations :exascript_r_tmp_cc) . && 
            python ./filter_swig_code.py "$@" "exascript_r_tmp.cc"
            """,
    outs = ["exascript_r.cc"],
    srcs = [":exascript_r_tmp_cc"],
    tools = ["//:filter_swig_code.py"]
)

cc_library(
    name = "exascript_r",
    srcs = [":filter_swig_code_exascript_r_cc",":filter_swig_code_exascript_r_h","//exaudflib:header"],
    hdrs = [":filter_swig_code_exascript_r_h"],
    copts = ["-DENABLE_R_VM"],
    deps = ["@R_linux//:r-lib","//exaudflib:exaudflib-deps"]
)

cc_library(
    name = "rcontainer",
    srcs = ["rcontainer.cc","//exaudflib:header"],
    copts = ["-DENABLE_R_VM"],
    include_prefix = ".",
    data = ["exascript_r_wrap.R", "exascript_r_preset.R"], #TODO do we need this, because build_integreted
    hdrs = ["filter_swig_code_exascript_r_h","exascript_r_int"],
    deps = ["@R_linux//:r-lib",":exascript_r"]
)