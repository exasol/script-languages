java_library(
    name = "ExaStackTraceCleaner",
    srcs = ["//base/javacontainer:ExaStackTraceCleaner.java"]
)

java_test(
    name = "ExaStackTraceCleanerTest",
    srcs = ["java/com/exasol/ExaStackTraceCleanerTest.java"],
    size = "small",
    deps = [":ExaStackTraceCleaner"]
)

JAVACONTAINER_TEST_SRCS = ["cpp/javacontainer_test.cc", "cpp/exaudf_wrapper.cc", "cpp/javavm_test.cc", "cpp/javavm_test.h",
            "cpp/swig_factory_test.h", "cpp/swig_factory_test.cc"]

cc_test(
    name = "javacontainer-test-legacy-parser",
    srcs = JAVACONTAINER_TEST_SRCS,
    deps = [
        "//base/javacontainer:javacontainer",
        "@googletest//:gtest_main",
    ],
    data = ["test.jar", "other_test.jar"]
)

cc_test(
    name = "javacontainer-test-ctpg-parser",
    srcs = JAVACONTAINER_TEST_SRCS,
    deps = [
        "//base/javacontainer:javacontainer",
        "@googletest//:gtest_main",
    ],
    defines = ["USE_CTPG_PARSER"],
    data = ["test.jar", "other_test.jar"]
)
