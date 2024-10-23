
cc_test(
    name = "java-script-options-tests",
    srcs = ["ctpg_script_importer_test.cc", "swig_factory_test.cc", "swig_factory_test.h", "string_ops_tests.cc"],
    deps = [
        "//base/javacontainer/script_options:java_script_option_lines",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "javacontainer-test-converter",
    srcs =  ["converter_test.cc"],
    deps = [
        "//base/javacontainer/script_options:java_script_option_lines",
        "@googletest//:gtest_main",
    ],
)
