package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "swig_factory_if",
    hdrs = [":swig_factory.h"],
)


cc_library(
    name = "swig_factory",
    hdrs = [":swig_factory.h", ":swig_factory_impl.h"],
    srcs = [":swig_factory_impl.cc"],
    deps = ["//exaudflib:header"],
)
