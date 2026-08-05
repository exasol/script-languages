package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "benchmark_container",
    srcs = ["benchmark_container.cc","benchmark_container.h"],
    hdrs = ["benchmark_container.h"],
    deps = ["//exaudflib:exaudflib-deps","//exaudflib:header", "//utils:utils"]+["//script_options_parser:scriptoptionlines"]
)