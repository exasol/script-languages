package(default_visibility = ["//visibility:public"])


cc_library(
    name = "java_script_option_lines",
    hdrs = [":extractor.h"],
    srcs = [":parser.h", ":converter.h", ":converter.cc", ":parser_legacy.h", ":parser_legacy.cc", ":extractor.cc"],
    deps = ["//base/script_options_parser:script_option_lines_parser", "//base/exaudflib:header", "//base/exaudflib:exaudflib-deps"],
)
