package(default_visibility = ["//visibility:public"])


cc_library(
    name = "swig_factory_if",
    hdrs = [":swig_factory.h"],
)


cc_library(
    name = "swig_factory",
    hdrs = [":swig_factory.h", ":swig_factory_impl.h"],
    srcs = [":swig_factory_impl.cc"],
    deps = ["//base/exaudflib:header"],
)
