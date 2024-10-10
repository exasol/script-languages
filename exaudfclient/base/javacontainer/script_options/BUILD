package(default_visibility = ["//visibility:public"])


cc_library(
    name = "java_script_option_lines",
    hdrs = [":extractor.h", ":parser_legacy.h"],
    srcs = [":parser.h", ":converter.h", ":converter.cc", ":parser_legacy.cc", ":extractor.cc",
            ":keywords.h", ":checksum.h", ":checksum.cc"],
    deps = ["//base/script_options_parser/legacy:script_option_lines_parser_legacy", "//base/utils:utils",
            "//base/exaudflib:header", "//base/exaudflib:exaudflib-deps", "//base/swig_factory:swig_factory_if",
            "//base/script_options_parser:exception"],
)
