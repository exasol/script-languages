package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "script_option_lines_parser_ctpg",
    hdrs = ["script_option_lines_ctpg.h"],
    srcs = ["script_option_lines_ctpg.cc","ctpg.hpp"],
    deps = ["//script_options_parser:exception"],
    copts= ["-fno-lto"],
)
