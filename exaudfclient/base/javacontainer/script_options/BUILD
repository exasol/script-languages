package(default_visibility = ["//visibility:public"])


cc_library(
    name = "java_script_option_lines",
    hdrs = [":extractor.h", ":parser_legacy.h", ":parser_ctpg.h"],
    srcs = [":parser.h", ":converter.h", ":converter.cc", ":parser_legacy.cc", ":extractor.cc",
            ":keywords.h", ":keywords.cc", ":checksum.h", ":checksum.cc", ":parser_ctpg.cc",
            ":parser_ctpg_script_importer.cc", ":parser_ctpg_script_importer.h"],
    deps = ["//base/script_options_parser/legacy:script_option_lines_parser_legacy",
            "//base/script_options_parser/ctpg:script_option_lines_parser_ctpg", "//base/utils:utils",
            "//base/exaudflib:header", "//base/exaudflib:exaudflib-deps", "//base/swig_factory:swig_factory_if",
            "//base/script_options_parser:exception"],
)
