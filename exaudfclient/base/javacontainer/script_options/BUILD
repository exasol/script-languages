package(default_visibility = ["//visibility:public"])


cc_library(
    name = "java_scriptoptionlines",
    hdrs = [":scriptoptionlinesconverter.h"],
    srcs = [":scriptoptionlinesparser.h", ":scriptoptionlinesconverter.h", ":scriptoptionlinesconverter.cc", ":scriptoptionlinesparserlegacy.h", ":scriptoptionlinesparserlegacy.cc"],
    deps = ["//base/script_options_parser:scriptoptionlinesparser", "//base/exaudflib:header", "//base/exaudflib:exaudflib-deps"],
)
