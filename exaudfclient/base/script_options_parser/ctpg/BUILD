package(default_visibility = ["//visibility:public"])

cc_library(
    name = "script_option_lines_parser_ctpg",
    hdrs = ["script_option_lines_ctpg.h"],
    srcs = ["script_option_lines_ctpg.cc","ctpg.hpp"],
    deps = ["//script_options_parser:exception"],
    copts= ["-fno-lto"],
)
