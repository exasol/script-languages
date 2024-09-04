package(default_visibility = ["//visibility:public"])


cc_library(
    name = "java_scriptoptionlines",
    hdrs = [":converter.h"],
    srcs = [":parser.h", ":converter.h", ":converter.cc", ":parser_legacy.h", ":parser_legacy.cc"],
    deps = ["//base/script_options_parser:scriptoptionlinesparser", "//base/exaudflib:header", "//base/exaudflib:exaudflib-deps"],
)
