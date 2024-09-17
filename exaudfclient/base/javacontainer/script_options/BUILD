package(default_visibility = ["//visibility:public"])


cc_library(
    name = "java_script_option_lines",
    hdrs = [":extractor.h", ":parser_factory.h"],
    srcs = [":parser.h", ":converter.h", ":converter.cc", ":parser_legacy.h", ":parser_legacy.cc", ":extractor.cc", ":parser_factory.cc"],
    deps = ["//base/script_options_parser:script_option_lines_parser", "//base/exaudflib:header", "//base/exaudflib:exaudflib-deps"],
)
