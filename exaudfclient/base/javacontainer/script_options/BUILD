package(default_visibility = ["//visibility:public"])


cc_library(
    name = "java_script_option_lines",
    hdrs = [":extractor.h", ":parser_legacy.h"],
    srcs = [":parser.h", ":converter.h", ":converter.cc", ":parser_legacy.cc", ":extractor.cc",
            ":keywords.h", ":checksum.h", ":checksum.cc"],
    deps = ["//base/script_options_parser:script_option_lines_parser", "//base:debug_message_h",
            "//base/exaudflib:header", "//base/exaudflib:exaudflib-deps"],
)
