package(default_visibility = ["//visibility:public"])


cc_library(
    name = "java_script_option_lines",
    hdrs = [":extractor.h", ":extractor_impl.h", ":parser_legacy.h", ":parser_ctpg.h",
            ":converter_v2.h", ":converter_legacy.h"],
    srcs = [":parser.h", ":converter.h", ":converter.cc", ":parser_legacy.cc", ":extractor_impl.h",
            ":extractor_impl.cc", ":converter_legacy.cc", ":converter_legacy.h",
            ":converter_v2.cc", ":converter_v2.h",
            ":keywords.h", ":keywords.cc", ":distinct_script_set.h", ":distinct_script_set.cc", ":parser_ctpg.cc",
            ":parser_ctpg_script_importer.cc", ":parser_ctpg_script_importer.h",
            ":parser_ctpg_jvm_options_parser.cc", ":parser_ctpg_jvm_options_parser.h", ":string_ops.h", ":string_ops.cc"],
    deps = ["//script_options_parser/legacy:script_option_lines_parser_legacy",
            "//script_options_parser/ctpg:script_option_lines_parser_ctpg", "//utils:utils",
            "//exaudflib:header", "//exaudflib:exaudflib-deps", "//swig_factory:swig_factory_if",
            "//script_options_parser:exception"],
)
