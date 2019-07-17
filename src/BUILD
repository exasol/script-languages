package(default_visibility = ["//visibility:public"])
exports_files(["filter_swig_code.py", "build_integrated.py","LICENSE-exasol-script-api.txt"])

config_setting(
    name = "benchmark",
    define_values = {"benchmark": "true"},
)

config_setting(
    name = "python",
    define_values = {"python": "true"},
)

config_setting(
    name = "r",
    define_values = {"r": "true"},
)

config_setting(
    name = "java",
    define_values = {"java": "true"},
)

config_setting(
    name = "bash",
    define_values = {"bash": "true"},
)

load("//:variables.bzl", "VM_ENABLED_DEFINES")

cc_library(
    name = "debug_message_h",
    hdrs = [
        "debug_message.h"
    ],
)

# Using cc_binary to build a shared library as output target is a workaround, because
# Bazel's cc_library are only intermediate stage and can be used as output target.
# It is necessary to include all dependencies of //exaudflib:exaudflib into deps and srcs,
# otherwise dlmopen won't find the symbols from these dependencies. The target //exaudflib:exaudflib-deps
# won't work either, because it only contains the libraries only as dependencies.
cc_binary(
    name = "libexaudflib_complete.so",
    linkshared = 1,
    srcs = ["//exaudflib:exaudflib", 
            "//exaudflib:zmqcontainer", 
            "//exaudflib:script_data_transfer_objects_wrapper", 
            "//exaudflib:script_data_transfer_objects"
            ],
    deps = [ "@zmq//:zmqlib", "@protobuf//:protobuf"],
    defines = VM_ENABLED_DEFINES
)

VM_ENABLED_DEPS=select({
        ":benchmark": ["//benchmark_container:benchmark_container"],
        "//conditions:default": []
    }) + select({
        ":r": ["//rcontainer:rcontainer"],
        "//conditions:default": []
    }) + select({
        ":java": ["//javacontainer:javacontainer"],
        "//conditions:default": []
    }) + select({
        ":bash": ["//streaming_container:streamingcontainer"],
        "//conditions:default": []
    })

VM_PYTHON2_DEPS=select({
        ":python": ["//python/python2:pythoncontainer"],
        "//conditions:default": []
    }) 

VM_PYTHON3_DEPS=select({
        ":python": ["//python/python3:pythoncontainer"],
        "//conditions:default": []
    }) 

cc_binary(
    name = "exaudfclient",
    srcs = ["exaudfclient.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON2_DEPS,
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"]
)

cc_binary(
    name = "exaudfclient_py3",
    srcs = ["exaudfclient.cc"],
    linkopts = ["-ldl"], # needed for dynamicly loading libexaudflib_complete.so into another linker namespace
    deps = ["//exaudflib:header", "//:debug_message_h"]+VM_ENABLED_DEPS+VM_PYTHON3_DEPS,
    defines = VM_ENABLED_DEFINES,
    data = [":libexaudflib_complete.so"]
)

# Workaround for the problems of JNI with rpath's used by Bazel
# Bazel creates for all external local repositories symlinks and 
# the path of these symlinks may contain a @ character.
# Bazel than write during compilation these paths to the rpath of the binary,
# but JNI truncates pathes with @ character. In the end, 
# it was nearly impossible to convince JNI from other pathes for its libraries, 
# as the original pathes where apt installed the files.
# genrule(
#     name = "gen-exaudfclient",
#     cmd = """
#         touch $(location exaudfclient)
#         cat $(location exaudfclient.sh) >> $(location exaudfclient)
#         echo >> $(location exaudfclient)
#         echo 'export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:'`find "$$JAVA_PREFIX" -name *.so -type f |  xargs -n 1 dirname | sort | uniq | paste -sd ":" -` >> $(location exaudfclient)
#         #echo 'echo "LD_LIBRARY_PATH: $$LD_LIBRARY_PATH"' >> $(location exaudfclient)
#         echo ./`basename $(location exaudfclient_wrong_rpath_py2)` '$$*' >>  $(location exaudfclient)
#     """,
#     outs = ["exaudfclient"],
#     srcs = [":exaudfclient_wrong_rpath_py2", "libexaudflib_complete.so", "exaudfclient.sh"],
#     #executable = True,
#     output_to_bindir = True
# )

# genrule(
#     name = "gen-exaudfclient-py3",
#     cmd = """
#         touch $(location exaudfclient_py3)
#         cat $(location exaudfclient.sh) >> $(location exaudfclient_py3)
#         echo >> $(location exaudfclient_py3)
#         echo 'export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:'`find "$$JAVA_PREFIX" -name *.so -type f |  xargs -n 1 dirname | sort | uniq | paste -sd ":" -` >> $(location exaudfclient_py3)
#         #echo 'echo "LD_LIBRARY_PATH: $$LD_LIBRARY_PATH"' >> $(location exaudfclient_py3)
#         echo ./`basename $(location exaudfclient_wrong_rpath_py3)` '$$*' >>  $(location exaudfclient_py3)
#     """,
#     outs = ["exaudfclient_py3"],
#     srcs = [":exaudfclient_wrong_rpath_py3", "libexaudflib_complete.so", "exaudfclient.sh"],
#     #executable = True,
#     output_to_bindir = True
# )

# Failed attempt to make java depdendency local independently to bazel, because JNI has problems with paths which contain a @. 
# However, JNI has problems with this attempt, too. 
# Unitl now only the path under /usr/lib/jvm worked.
#
# genrule(
#     name = "gen-java-lib",
#     cmd = """
#         mkdir $(location java-lib)
#         cd $(location java-lib)
#         LIB_NAMES=`find $$JAVA_PREFIX -name '*.so' | xargs -n 1 basename | sort | uniq`
#         #echo LIB_NAMES:  $$LIB_NAMES
#         for I in $$LIB_NAMES
#         do
#             FILE_TO_LINK=`find $$JAVA_PREFIX -name $$I -type f | head -n 1`
#             #echo FILE_TO_LINK: $$FILE_TO_LINK
#             ln -s $$FILE_TO_LINK
#         done
#     """,
#     outs = ["java-lib"],
#     output_to_bindir = True
# )