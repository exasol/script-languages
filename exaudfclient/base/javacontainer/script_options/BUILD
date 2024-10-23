package(default_visibility = ["//visibility:public"])


cc_library(
    name = "java_script_option_lines",
    hdrs = [":extractor.h", ":extractor_impl.h", ":parser_legacy.h", ":parser_ctpg.h",
            ":converter_v2.h", ":converter_legacy.h"],
    srcs = [":parser.h", ":converter.h", ":converter.cc", ":parser_legacy.cc", ":extractor_impl.h",
            ":extractor_impl.cc", ":converter_legacy.cc", ":converter_legacy.h",
            ":converter_v2.cc", ":converter_v2.h",
            ":keywords.h", ":keywords.cc", ":checksum.h", ":checksum.cc", ":parser_ctpg.cc",
            ":parser_ctpg_script_importer.cc", ":parser_ctpg_script_importer.h", ":string_ops.h", ":string_ops.cc"],
    deps = ["@ssl//:ssl", "//base/script_options_parser/legacy:script_option_lines_parser_legacy",
            "//base/script_options_parser/ctpg:script_option_lines_parser_ctpg", "//base/utils:utils",
            "//base/exaudflib:header", "//base/exaudflib:exaudflib-deps", "//base/swig_factory:swig_factory_if",
            "//base/script_options_parser:exception"],
)
